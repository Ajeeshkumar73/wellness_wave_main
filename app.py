from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, send_file
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from werkzeug.utils import secure_filename
import os, io
import pandas as pd
import numpy as np
import joblib
from flask_socketio import SocketIO, emit, join_room
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import pytz
import json
from pytorch_tabnet.tab_model import TabNetRegressor

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether)
from reportlab.graphics.shapes import Drawing, Rect, String


from bot import lifestyle_disease_chat
from precaution import ai_precautions_groq

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 MUST be defined BEFORE allowed_file()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "docx"}

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


BLOG_UPLOAD_FOLDER = "static/blogs"
os.makedirs(BLOG_UPLOAD_FOLDER, exist_ok=True)

app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="wellnesswave353@gmail.com",
    MAIL_PASSWORD=os.environ.get("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER="Wellness Wave <wellnesswave353@gmail.com>"
)

mail = Mail(app)

ADMIN_EMAIL = 'wellnesswave353@gmail.com'

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["healthcare_db"]
users_col = db["users"]
doctors_col = db["doctors"]
appointments_col = db["appointments"]
predictions_col = db["predictions"]
messages_col = db["messages"]
reviews_col = db["reviews"]
analysis_collection = db["analyses"]
blog_col = db["blogs"]
chat_col = db["chats"]
notifications_col = db.notifications

# ── Computed features (auto-calculated, not user input) ───────────────────────
COMPUTED_FEATURES = {"BMI", "Body Fat %"}

# ── Binary checkbox features — only include if user checked them (value = 1) ──
BINARY_CHECKBOX_FEATURES = {
    "Family History: Diabetes",
    "Family History: Heart Disease",
    "Existing BP Issues",
    "Shortness of Breath",
    "Frequent Urination",
    "Excessive Thirst",
}

NEUTRAL_VALUES = {
    "Smoking":            {"Never"},
    "Alcohol_Consumption": {"None"},
    "Salt_intake":         {"Low"},
    "Sleep_quality":       {"Good"},
    "Junk_food_per_week":  {0},
    "Red_meat_per_week":   {0},
    "Fried_food_per_week": {0},
}

# ── Maps FEATURE_NAME → (input_data_key, display_label, unit) ─────────────────
FEATURE_META = {
    "Age":                            ("Age",                              "Age",                    "yrs"),
    "Gender":                         ("Gender",                           "Gender",                 ""),
    "Height (cm)":                    ("Height_cm",                        "Height",                 "cm"),
    "Weight (kg)":                    ("Weight_kg",                        "Weight",                 "kg"),
    "Family History: Diabetes":       ("Family_History_Diabetes",          "Family Hx: Diabetes",    ""),
    "Family History: Heart Disease":  ("Family_History_Heart_Disease",     "Family Hx: Heart",       ""),
    "Existing BP Issues":             ("Existing_BP_Issues",               "Existing BP Issues",     ""),
    "Cholesterol (mg/dL)":            ("Cholesterol_mg_dL",                "Cholesterol",            "mg/dL"),
    "Physical Activity (days/wk)":    ("Physical_Activity_days_per_week",  "Physical Activity",      "days/wk"),
    "Smoking":                        ("Smoking",                          "Smoking",                ""),
    "Sleep Hours":                    ("Sleep_hours",                      "Sleep",                  "hrs"),
    "Junk Food (per wk)":             ("Junk_food_per_week",               "Junk Food",              "/wk"),
    "Salt Intake":                    ("Salt_intake",                      "Salt Intake",            ""),
    "Systolic BP":                    ("Systolic_BP",                      "Systolic BP",            "mmHg"),
    "Diastolic BP":                   ("Diastolic_BP",                     "Diastolic BP",           "mmHg"),
    "Shortness of Breath":            ("Shortness_of_breath",              "Shortness of Breath",    ""),
    "Waist Circumference (cm)":       ("Waist_circumference_cm",           "Waist",                  "cm"),
    "Frequent Urination":             ("Frequent_urination",               "Frequent Urination",     ""),
    "Excessive Thirst":               ("Excessive_thirst",                 "Excessive Thirst",       ""),
    "Sleep Quality":                  ("Sleep_quality",                    "Sleep Quality",          ""),
    "Red Meat (per wk)":              ("Red_meat_per_week",                "Red Meat",               "/wk"),
    "Fried Food (per wk)":            ("Fried_food_per_week",              "Fried Food",             "/wk"),
    "Water Intake (L/day)":           ("Water_intake_liters_per_day",      "Water Intake",           "L/day"),
    "Alcohol Consumption":            ("Alcohol_Consumption",              "Alcohol",                ""),
    "Blood Sugar (mg/dL)":            ("Blood_Sugar_mg_dL",                "Blood Sugar",            "mg/dL"),
}

# ── Categorical reverse-decode map (encoded int → original string) ─────────────
# Used to show "Never" instead of "0" for Smoking, etc.
CATEGORICAL_KEYS = {
    "Smoking", "Salt_intake", "Sleep_quality", "Alcohol_Consumption", "Gender"
}


def _decode_value(input_key, raw_input_data_before_encode, label_encoders):
    """Return the human-readable value for a feature."""
    # For binary checkboxes
    val = raw_input_data_before_encode.get(input_key)
    if val in (0, 1):
        return "Yes" if val == 1 else "No"
    return str(val) if val is not None else "—"

# import models
def load_tabnet(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    model = TabNetRegressor()
    model.load_model(path)
    return model

models = {
    "Diabetes": load_tabnet("Diabetes_Risk_%_tabnet_model.zip"),
    "Heart_Disease": load_tabnet("Heart_Disease_Risk_%_tabnet_model.zip"),
    "Hypertension": load_tabnet("Hypertension_Risk_%_tabnet_model.zip"),
    "Obesity": load_tabnet("Obesity_Risk_%_tabnet_model.zip")
}

label_encoders = joblib.load("label_encoders.pkl")


# ─── Status helpers ───────────────────────────────────────────────────────────
#bmistatus
def bmi_status(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

#blood pressure status
def bp_status(sys, dia):
    if sys < 120 and dia < 80:
        return "Normal"
    elif sys < 140 or dia < 90:
        return "Intermediate"
    else:
        return "High"

# blood suger status
def sugar_status(val):
    if val < 100:
        return "Normal"
    elif val < 126:
        return "Intermediate"
    else:
        return "High"
    
#cholesterol status
def cholesterol_status(val):
    if val < 200:
        return "Normal"
    elif val < 240:
        return "Intermediate"
    else:
        return "High"

# ─── TabNet explainability (CORRECT implementation) ───────────────────────────
def get_feature_importance(model, features_array):

    explain_matrix, _ = model.explain(features_array)
    importance = explain_matrix[0]

    total = importance.sum()

    if total > 0:
        pcts = (importance / total * 100)
    else:
        pcts = [100.0 / len(FEATURE_META)] * len(FEATURE_META)

    paired = [
        (fname, float(pct))
        for fname, pct in zip(FEATURE_META, pcts)
        if pct > 0.01        # ← removes 0 and near-zero values
    ]

    paired.sort(key=lambda x: x[1], reverse=True)

    return paired


#prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.form

    height = float(data['height'])
    weight = float(data['weight'])
    bmi = round(weight / ((height / 100) ** 2), 2)
    
    gender_flag = 1 if data['gender'] == "Male" else 0
    age = int(data['age'])

    body_fat = round(
        (1.20 * bmi) + (0.23 * age) - (10.8 * gender_flag) - 5.4,2
    )
   

    input_data = {
        "Age":                             age,
        "Gender":                          data["gender"],
        "Height_cm":                       height,
        "Weight_kg":                       weight,
        "Body_fat_percent":                body_fat,
        "Family_History_Diabetes":         int(data.get("family_diabetes") == "on"),
        "Family_History_Heart_Disease":    int(data.get("family_heart") == "on"),
        "Existing_BP_Issues":              int(data.get("existing_bp") == "on"),
        "Shortness_of_breath":             int(data.get("breath") == "on"),
        "Frequent_urination":              int(data.get("urination") == "on"),
        "Excessive_thirst":                int(data.get("thirst") == "on"),
        "Cholesterol_mg_dL":               float(data["cholesterol"]),
        "Physical_Activity_days_per_week": int(data["activity"]),
        "Sleep_hours":                     float(data["sleep"]),
        "Junk_food_per_week":              int(data["junk"]),
        "Systolic_BP":                     float(data["systolic"]),
        "Diastolic_BP":                    float(data["diastolic"]),
        "Waist_circumference_cm":          float(data["waist"]),
        "Red_meat_per_week":               int(data["redmeat"]),
        "Fried_food_per_week":             int(data["fried"]),
        "Water_intake_liters_per_day":     float(data["water"]),
        "Blood_Sugar_mg_dL":               float(data.get("bloodSugar", 0)),
        "Smoking":                         data["smoking"],
        "Salt_intake":                     data["salt"],
        "Sleep_quality":                   data["sleep_quality"],
        "Alcohol_Consumption":             data["alcohol"],
        "BMI":                             bmi,
    }
    

    disease_results, explain_results, _  = predict_risk(input_data)

    metrics = {
        "BMI": {"value": bmi, "status": bmi_status(bmi)},
        "BloodPressure": {
            "value": f"{data['systolic']}/{data['diastolic']}",
            "status": bp_status(float(data['systolic']), float(data['diastolic']))
        },
        "BloodSugar": {
            "value": float(data['bloodSugar']),
            "status": sugar_status(float(data['bloodSugar']))
        },
        "Cholesterol": {
            "value": float(data['cholesterol']),
            "status": cholesterol_status(float(data['cholesterol']))
        }
    }

    # 🔹 GROQ AI PRECAUTIONS
    precautions_raw = ai_precautions_groq(metrics, disease_results,input_data)

    if isinstance(precautions_raw, dict):
        precautions = precautions_raw          # new format: {type, diet, exercise, habits}
    elif isinstance(precautions_raw, str):
        try:
            precautions = json.loads(precautions_raw)
        except json.JSONDecodeError:
            precautions = []
    elif isinstance(precautions_raw, list):
        precautions = precautions_raw
    else:
        precautions = []

        # 🔹 Collect numeric risk values
    risk_values = [
        v["risk_percentage"]
        for k, v in disease_results.items()
    ]

    # 🔹 Overall health score (higher = better)
    avg_risk = float(np.mean(risk_values))
    health_score = round(100 - avg_risk, 2)

    # 🔹 Health status classification
    if health_score >= 75:
        health_status = "Good Standing"
    elif health_score >= 50:
        health_status = "Moderate Risk"
    else:
        health_status = "High Risk"

    # 🔹 Add to results
    disease_results["overall_health"] = {
        "health_score": health_score,
        "status": health_status
    }

    payload = {
        "metrics": metrics, 
        "diseases": disease_results,
        "overall_health": disease_results["overall_health"],
        "precautions": precautions, 
        "explain": explain_results,
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
    }
    session["last_report"] = payload


    analysis_collection.insert_one({
        "user_id": session["user_id"],
        "metrics": metrics,
        "diseases": disease_results,
        "precautions": precautions,
        "explain": {k: v.get("top5",[]) for k,v in explain_results.items()},
        "datetime": datetime.now()
    })

    return jsonify(payload)

@app.route("/download_report")
def download_report():
    if "user_id" not in session: return redirect(url_for("login"))
    payload = session.get("last_report")
    if not payload: return "No report available. Run an analysis first.", 400
    buf   = build_pdf_report(payload)
    fname = f"Wellnesswave_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/pdf")

# anlaysis by deeping model
def predict_risk(input_data):
    # Encode categorical values
    raw_input = dict(input_data)

    categorical_cols = [
        "Gender",
        "Smoking",
        "Salt_intake",
        "Sleep_quality",
        "Alcohol_Consumption"
    ]

        # Encode categorical
    for col in categorical_cols:
        encoder = label_encoders[col]

        if input_data[col] not in encoder.classes_:
            input_data[col] = encoder.classes_[0]

        input_data[col] = int(encoder.transform([input_data[col]])[0])


    # 27 features — EXACT same order as train_model.py
    features = np.array([[
        input_data["Age"],                              # 0
        input_data["Gender"],                           # 1
        input_data["Height_cm"],                        # 2
        input_data["Weight_kg"],                        # 3
        input_data["Family_History_Diabetes"],          # 4
        input_data["Family_History_Heart_Disease"],     # 5
        input_data["Existing_BP_Issues"],               # 6
        input_data["Cholesterol_mg_dL"],                # 7
        input_data["Physical_Activity_days_per_week"],  # 8
        input_data["Smoking"],                          # 9
        input_data["Sleep_hours"],                      # 10
        input_data["Junk_food_per_week"],               # 11
        input_data["Salt_intake"],                      # 12
        input_data["Systolic_BP"],                      # 13
        input_data["Diastolic_BP"],                     # 14
        input_data["Shortness_of_breath"],              # 15
        input_data["Waist_circumference_cm"],           # 16
        input_data["Frequent_urination"],               # 17
        input_data["Excessive_thirst"],                 # 18
        input_data["Body_fat_percent"],                 # 19
        input_data["Sleep_quality"],                    # 20
        input_data["Red_meat_per_week"],                # 21
        input_data["Fried_food_per_week"],              # 22
        input_data["Water_intake_liters_per_day"],      # 23
        input_data["BMI"],                              # 24
        input_data["Alcohol_Consumption"],              # 25
        input_data["Blood_Sugar_mg_dL"],                # 26
    ]], dtype=np.float32)

    assert features.shape == (1, 27), f"Feature shape error: {features.shape}"


    results = {}
    explain_results = {}


    for disease, model in models.items():
        prediction = float(model.predict(features)[0][0])

        # Clamp between 0–100
        prediction = max(0, min(100, prediction))

        if prediction >= 85:
            risk = "High"
        elif prediction >= 65:
            risk = "Borderline"    
        elif prediction >= 30:
            risk = "Intermediate"
        else:
            risk = "Low"


        results[disease] = {
            "risk_percentage": round(prediction, 2),
            "risk_level": risk,
        }

        # ── Explainability ────────────────────────────────────────────────────
        try:
            all_importance = get_feature_importance(model, features)
            # → [(feature_name, pct), ...] sorted desc

            filtered = []
            for feat_name, pct in all_importance:

                # Skip computed features
                if feat_name in COMPUTED_FEATURES:
                    continue

                # Skip features not in our user-input map
                if feat_name not in FEATURE_META:
                    continue

                input_key, display_label, unit = FEATURE_META[feat_name]

                # Skip binary checkboxes the user did NOT tick
                if feat_name in BINARY_CHECKBOX_FEATURES:
                    if raw_input.get(input_key, 0) == 0:
                        continue

                user_raw_val = raw_input.get(input_key)
                if input_key in NEUTRAL_VALUES:
                    if user_raw_val in NEUTRAL_VALUES[input_key]:
                        continue

                # Get the human-readable value the user entered
                user_val = raw_input.get(input_key)
                if user_val in (0, 1) and feat_name in BINARY_CHECKBOX_FEATURES:
                    display_val = "Yes" if user_val == 1 else "No"
                else:
                    display_val = str(user_val) if user_val is not None else "—"

                filtered.append({
                    "name":        feat_name,       # original feature name
                    "label":       display_label,   # short friendly label
                    "value":       display_val,     # what the user entered
                    "unit":        unit,            # e.g. "mg/dL", "yrs"
                    "pct":         pct,             # raw importance weight
                })

            # Re-normalise percentages to sum to 100%
            total = sum(f["pct"] for f in filtered)
            if total > 0:
                for f in filtered:
                    f["pct"] = round(f["pct"] / total * 100, 2)

            # Sort descending by contribution
            filtered.sort(key=lambda x: x["pct"], reverse=True)

            # Serialise as list of tuples for top5 (backward compat with frontend)
            # and full list of dicts for all27 (new richer format)
            explain_results[disease] = {
                "top5":  [(f["label"], f["pct"], f["value"], f["unit"]) for f in filtered[:5]],
                "all27": [(f["label"], f["pct"], f["value"], f["unit"]) for f in filtered],
            }

        except Exception as e:
            explain_results[disease] = {
                "top5":  [], "all27": [], "error": str(e)
            }

    return results, explain_results, features

def build_pdf_report(payload):
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title="Wellness Wave Health Risk Report"
    )

    W = A4[0] - 4*cm

    def S(name, **kw): return ParagraphStyle(name, **kw)

    sTitle  = S("T",  fontSize=20, fontName="Helvetica-Bold",
                alignment=TA_CENTER, spaceAfter=6)
    sSub    = S("Su", fontSize=10, fontName="Helvetica",
                alignment=TA_CENTER, spaceAfter=4)
    sH1     = S("H1", fontSize=13, fontName="Helvetica-Bold",
                spaceBefore=16, spaceAfter=8,
                borderPad=4, backColor=colors.HexColor("#f0f4f8"))
    sBody   = S("B",  fontSize=9,  fontName="Helvetica",
                spaceAfter=3, leading=13)
    sBold   = S("Bd", fontSize=9,  fontName="Helvetica-Bold",
                spaceAfter=3)
    sCenter = S("C",  fontSize=28, fontName="Helvetica-Bold",
                alignment=TA_CENTER, spaceAfter=4)
    sStatus = S("St", fontSize=12, fontName="Helvetica-Bold",
                alignment=TA_CENTER, spaceAfter=6)
    sFooter = S("F",  fontSize=7,  fontName="Helvetica",
                alignment=TA_CENTER, textColor=colors.grey)

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#cccccc"),
                          spaceAfter=10, spaceBefore=6)

    def risk_color(level):
        return {
            "High":         colors.HexColor("#ff5e57"),
            "Borderline":   colors.HexColor("#ff7f50"),
            "Intermediate": colors.HexColor("#ff9f43"),
            "Low":          colors.HexColor("#00c896"),
        }.get(level, colors.grey)

    def status_color(status):
        sl = (status or "").lower()
        if sl == "normal":                          return colors.HexColor("#00c896")
        if sl in ("intermediate", "overweight"):    return colors.HexColor("#ff9f43")
        return colors.HexColor("#ff5e57")

    metrics  = payload["metrics"]
    diseases = payload["diseases"]
    explain  = payload.get("explain", {})
    prec     = payload.get("precautions", {})
    oh       = payload["overall_health"]
    gen      = payload.get("generated_at",
                           datetime.now().strftime("%d %b %Y, %H:%M"))

    story = []

    # ═══════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════
    story += [
        Spacer(1, .3*cm),
        Paragraph("Wellness Wave", sTitle),
        Paragraph("Health Risk Assessment Report", sSub),
        Paragraph(f"Generated: {gen}", sSub),
        Spacer(1, .4*cm),
        hr(),
    ]

    # ═══════════════════════════════════════════════
    # OVERALL HEALTH SCORE
    # ═══════════════════════════════════════════════
    score  = oh.get("health_score", 0)
    status = oh.get("status", "")
    sc     = "#00c896" if score >= 75 else ("#ff9f43" if score >= 50 else "#ff5e57")

    story += [
        Paragraph("OVERALL HEALTH SCORE", sH1),
        Paragraph(
            f'<font color="{sc}">{score}%</font>',
            sCenter
        ),
        Spacer(1, 14),
        Paragraph(
            f'<font color="{sc}">{status}</font>',
            sStatus
        ),
        Spacer(1, .5*cm),
        hr(),
    ]

    # ═══════════════════════════════════════════════
    # VITAL METRICS TABLE
    # ═══════════════════════════════════════════════
    story += [Paragraph("VITAL METRICS", sH1)]

    mrows = [[
        Paragraph("<b>Metric</b>", sBold),
        Paragraph("<b>Value</b>", sBold),
        Paragraph("<b>Status</b>", sBold),
    ]]
    for lbl, key in [
        ("BMI",                "BMI"),
        ("Blood Pressure",     "BloodPressure"),
        ("Blood Sugar (mg/dL)","BloodSugar"),
        ("Cholesterol (mg/dL)","Cholesterol"),
    ]:
        m  = metrics.get(key, {})
        st = str(m.get("status", "—"))
        mrows.append([
            Paragraph(lbl, sBody),
            Paragraph(f"<b>{m.get('value','—')}</b>", sBold),
            Paragraph(
                f'<font color="{status_color(st).hexval() if hasattr(status_color(st),"hexval") else "#000"}">{st}</font>',
                sBody
            ),
        ])

    mt = Table(mrows, colWidths=[W*.42, W*.28, W*.30])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#e8f0fe")),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("ALIGN",         (1,1), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),
         [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    story += [mt, Spacer(1, .6*cm), hr()]

    # ═══════════════════════════════════════════════
    # DISEASE RISK TABLE
    # ═══════════════════════════════════════════════
    story += [Paragraph("DISEASE RISK ANALYSIS", sH1)]

    labels = {
        "Diabetes":     "Diabetes",
        "Heart_Disease":"Heart Disease",
        "Hypertension": "Hypertension",
        "Obesity":      "Obesity",
    }

    drows = [[
        Paragraph("<b>Disease</b>",    sBold),
        Paragraph("<b>Risk %</b>",     sBold),
        Paragraph("<b>Risk Level</b>", sBold),
    ]]
    for key, lbl in labels.items():
        d = diseases.get(key)
        if not d: continue
        level = d["risk_level"]
        rc    = risk_color(level)
        drows.append([
            Paragraph(lbl, sBody),
            Paragraph(f"<b>{d['risk_percentage']:.1f}%</b>", sBold),
            Paragraph(
                f'<font color="#{rc.hexval() if hasattr(rc,"hexval") else "000000"}">'
                f'<b>{level}</b></font>',
                sBody
            ),
        ])

    dt = Table(drows, colWidths=[W*.42, W*.28, W*.30])
    dt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#e8f0fe")),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("ALIGN",         (1,1), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),
         [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    story += [dt, Spacer(1, .6*cm), hr()]

    # ═══════════════════════════════════════════════
    # EXPLAINABILITY — KEY RISK FACTORS
    # ═══════════════════════════════════════════════
    story += [Paragraph("MODEL EXPLAINABILITY — KEY RISK FACTORS", sH1)]
    story += [
        Paragraph(
            "The table below shows which factors from your input contributed most "
            "to each disease risk prediction, along with the value you provided.",
            sBody
        ),
        Spacer(1, .3*cm),
    ]

    for disease_key, disease_label in labels.items():

        exp          = explain.get(disease_key, {})
        all_features = exp.get("all27", [])
        if not all_features:
            continue

        d     = diseases.get(disease_key, {})
        level = d.get("risk_level", "Low")
        rc    = risk_color(level)

        story += [
            Spacer(1, .2*cm),
            Paragraph(
                f'<b>{disease_label}</b> — '
                f'Risk: <b>{d.get("risk_percentage", 0):.1f}%</b> | '
                f'Level: <b>{level}</b>',
                sBold
            ),
            Spacer(1, .1*cm),
        ]

        erows = [[
            Paragraph("<b>#</b>",            sBold),
            Paragraph("<b>Factor</b>",       sBold),
            Paragraph("<b>Your Value</b>",   sBold),
            Paragraph("<b>Contribution</b>", sBold),
        ]]

        for i, feat_item in enumerate(all_features, 1):
            # 4-tuple: (label, pct, user_val, unit)  ← new format
            # 2-tuple: (label, pct)                  ← old format fallback
            if len(feat_item) == 4:
                fname, fpct, user_val, unit = feat_item
                display_val = f"{user_val} {unit}".strip()
            else:
                fname, fpct = feat_item
                display_val = "—"

            erows.append([
                Paragraph(str(i),              sBody),
                Paragraph(str(fname),          sBody),
                Paragraph(str(display_val),    sBody),
                Paragraph(f"{fpct:.2f}%",      sBold),
            ])

        et = Table(erows, colWidths=[W*.06, W*.42, W*.28, W*.24])
        et.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#e8f0fe")),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("ALIGN",         (3,1), (3,-1),  "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [colors.white, colors.HexColor("#f8f9fa")]),
        ]))

        story += [et, Spacer(1, .5*cm)]

    story += [hr()]


    # ═══════════════════════════════════════════════
    # DISCLAIMER
    # ═══════════════════════════════════════════════
    story += [
        Spacer(1, .4*cm),
        Paragraph(
            "DISCLAIMER: This report is generated by an AI system and is for "
            "informational purposes only. It does not constitute medical advice, "
            "diagnosis, or treatment. Always consult a qualified healthcare "
            "professional for any health concerns.",
            sFooter
        ),
    ]

    doc.build(story)
    buf.seek(0)
    return buf

# Home page
@app.route("/")
def index():
    latest_blogs = list(
        blog_col.find().sort("_id", -1).limit(3)
    )
    reviews = list(reviews_col.find().sort("created_at", -1))
    open_login = request.args.get("open_login", "false")
    return render_template("index.html", open_login=open_login, reviews=reviews, latest_blogs=latest_blogs)

# User registration
@app.route("/user_register", methods=["POST"])
def user_register():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")


    if users_col.find_one({"username": username}):
        flash("Username already taken. Try a different one.", "error")
        return redirect(url_for("index"))

    if users_col.find_one({"email": email}):
        flash("Email already registered. Try login instead.", "error")
        return redirect(url_for("index"))

    # Save user
    users_col.insert_one({
        "username": username,
        "email": email,
        "password": generate_password_hash(password),
        "created_at": datetime.now()
    })

    flash("Registration successful! Please login.", "success")
    return redirect(url_for("user_loginpage", open_login="true"))

# User login
@app.route("/user_login", methods=["POST"])
def user_login():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    user = users_col.find_one({"email": email})

    if user and check_password_hash(user.get("password", ""), password):

        #session.pop("doctor_id", None)   # clear doctor session
        #session.pop("user_type", None)   # clear old role (optional but safe)

        session["email"] = email
        session["user_id"] = str(user["_id"])
        session["user_type"] = "user"
        flash("Login successful!", "success")
        return redirect(url_for("user_home"))
    else:
        flash("Invalid email or password.", "error")
        return redirect(url_for("user_loginpage"))
    
@app.route('/user_loginpage')
def user_loginpage():
    return render_template("users_login.html")

@app.route('/user_reg')
def user_reg():
    return render_template("users_register.html")

    
# ---------------- ADMIN VIEW USERS ----------------
@app.route("/admin_page")
def admin_page():
    users = list(users_col.find().sort("created_at", -1))
    doctors = list(doctors_col.find().sort("created_at", -1))

    # Normalize created_at for users and doctors
    def normalize_created_at(record):
        if "created_at" in record:
            if isinstance(record["created_at"], str):
                try:
                    record["created_at"] = datetime.fromisoformat(record["created_at"])
                except ValueError:
                    record["created_at"] = None
        else:
            record["created_at"] = None
        return record

    users = [normalize_created_at(u) for u in users]
    doctors = [normalize_created_at(d) for d in doctors]

    return render_template("admin_page.html", users=users, doctors=doctors,  user_count=len(users),
        doctor_count=len(doctors),
        pending_doctor_count=len([d for d in doctors if not d.get("approved")]))


# ---------------- DELETE USER ----------------
@app.route("/admin/delete_user/<user_id>", methods=["POST"])
def delete_user(user_id):
    try:
        users_col.delete_one({"_id": ObjectId(user_id)})
        flash("User deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting user: {str(e)}", "error")
    return redirect(url_for("admin_page", section="user"))  # FIXED
 

# User dashboard
@app.route("/user_home")
def user_home():
    if 'user_id' not in session:
        return redirect(url_for("index"))

    user = users_col.find_one({"_id": ObjectId(session["user_id"])})

    recent_predictions = list(predictions_col.find(
        {"user_id": ObjectId(session["user_id"])}
    ).sort("created_at", -1).limit(5))

    upcoming_appointments = list(appointments_col.find({
        "user_id": ObjectId(session["user_id"]),
        "appointment_date": {"$gte": datetime.now()}
    }))

    return render_template(
        "user_home.html",
        user=user,
        recent_predictions=recent_predictions,
        upcoming_appointments=upcoming_appointments
    )

@app.route('/predict_page')
def predict_page():
    return render_template('predict_page.html')  # your prediction form page

@app.route('/back_page')
def back_page():
    return redirect(url_for('user_home'))

@app.route("/doctor_reg")
def doctor_reg():
    return render_template("doctor_register.html")

@app.route("/doctor_log")
def doctor_log():
    return render_template("doctor_login.html")


@app.route('/doctor_register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        specialization = request.form.get("specialization", "").strip()
        qualification = request.form.get("qualification", "").strip()
        experience = request.form.get("experience", "").strip()
        registration_id = request.form.get("registration_id", "").strip()
        hospital = request.form.get("hospital", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        password = request.form.get("password", "")

        # Handle profile picture
        profile_pic = request.files.get("profile_pic")
        profile_pic_filename = None
        if profile_pic and profile_pic.filename != "":
            filename = secure_filename(profile_pic.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            profile_pic.save(save_path)
            profile_pic_filename = f"uploads/{filename}"

        # Prevent duplicate
        if doctors_col.find_one({"email": email}):
            flash("Doctor email already registered.", "error")
            return redirect(url_for("index"))

        # Insert into DB
        doctors_col.insert_one({
            "name": name,
            "email": email,
            "specialization": specialization,
            "qualification": qualification,
            "experience": experience,
            "registration_id": registration_id,
            "hospital": hospital,
            "phone": phone,
            "address": address,
            "profile_pic": profile_pic_filename,
            "password": generate_password_hash(password),
            "approved": False,   # Needs admin approval
            "created_at": datetime.now()
        })

        # Send email to admin
        try:
            msg = Message(
                subject="New Doctor Registration",
                sender=app.config['MAIL_USERNAME'],
                recipients=[ADMIN_EMAIL],
                body=f"""
Hello Admin of Wellness Wave,

A new doctor has registered on Wellness Wave.

Name: {name}
Email: {email}
Specialization: {specialization}
Qualification: {qualification}
Experience: {experience} years
Registration ID: {registration_id}
Hospital: {hospital}
Phone: {phone}
Address: {address}

Please review and approve the account.
"""
            )
            mail.send(msg)
        except Exception as e:
            print("Failed to send admin email:", e)

        flash("Doctor registered successfully! Waiting for admin approval.", "success")
        return redirect(url_for("index", open_login="true"))

    return render_template("doctor_register.html")



@app.route("/doctor_loginpage")
def doctor_loginpage():
    return render_template("doctor_login.html")


#----------Approve/reject------------
@app.route("/admin/approve_doctor/<doctor_id>", methods=["POST"])
def approve_doctor(doctor_id):
    try:
        doctors_col.update_one({"_id": ObjectId(doctor_id)}, {"$set": {"approved": True}})
        flash("Doctor approved successfully.", "success")
    except Exception as e:
        flash(f"Error approving doctor: {str(e)}", "error")
    return redirect(url_for("admin_page", section="doctor"))


@app.route("/admin/reject_doctor/<doctor_id>", methods=["POST"])
def reject_doctor(doctor_id):
    try:
        doctors_col.delete_one({"_id": ObjectId(doctor_id)})
        flash("Doctor rejected and removed.", "success")
    except Exception as e:
        flash(f"Error rejecting doctor: {str(e)}", "error")
    return redirect(url_for("admin_page", section="doctor"))
   
    

#doctor login
@app.route('/doctor_login', methods=['GET', 'POST'])
def doctor_login():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    doctor = doctors_col.find_one({"email": email})

    if not doctor:
        flash("Doctor not found.", "error")
        return redirect(url_for("index", open_login="true"))

    if not doctor.get("approved", False):
        flash("Your account is pending admin approval.", "warning")
        return redirect(url_for("index", open_login="true"))

    if not check_password_hash(doctor["password"], password):
        flash("Invalid password.", "error")
        return redirect(url_for("index", open_login="true"))
    
    #session.pop("user_id", None)      # clear patient session
    #session.pop("user_type", None)    # clear role (optional but safe)

    # Save doctor session
    session["doctor_id"] = str(doctor["_id"])
    flash("Login successful!", "success")
    return redirect(url_for("doctor_home"))

#doctor dashboard
@app.route('/doctor_home', methods=['GET', 'POST'])
def doctor_home():
    # Ensure doctor is logged in
    if 'doctor_id' not in session:
        return redirect(url_for('doctor_login'))

    doctor_id = ObjectId(session['doctor_id'])
    doctor = doctors_col.find_one({'_id': doctor_id})
    if not doctor:
        flash("Doctor not found.", "error")
        return redirect(url_for('doctor_login'))

    # Handle profile update
    if request.method == "POST":
        update_data = {
            "name": request.form.get("name", "").strip(),
            "specialization": request.form.get("specialization", "").strip(),
            "qualification": request.form.get("qualification", "").strip(),
            "experience": request.form.get("experience", "").strip(),
            "registration_id": request.form.get("registration_id", "").strip(),
            "hospital": request.form.get("hospital", "").strip(),
            "phone": request.form.get("phone", "").strip(),
            "address": request.form.get("address", "").strip(),
            "bio": request.form.get("bio", "").strip(),
        }

        # Profile picture update
        profile_pic = request.files.get("profile_pic")
        if profile_pic and profile_pic.filename != "":
            filename = secure_filename(profile_pic.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            profile_pic.save(save_path)
            update_data["profile_pic"] = f"uploads/{filename}"

        # Update doctor in DB
        doctors_col.update_one(
            {"_id": doctor_id},
            {"$set": update_data}
        )
        flash("Profile updated successfully.", "success")
        return redirect(url_for("doctor_home"))

    # Convert _id for frontend
    doctor["_id"] = str(doctor["_id"])

    # Fetch appointments sorted by date descending
    
    appointments = list(appointments_col.find({'doctor_id': doctor_id}).sort('appointment_date', -1))
    

    # Attach patient info
    for appt in appointments:
        appt["_id"] = str(appt["_id"])

        user = users_col.find_one({"_id": appt["user_id"]})
        appt["user"] = user

     
    return render_template(
        'doctor_home.html',
        doctor=doctor,
        user=user,
        appointments=appointments
        
    )

@app.route("/add_review", methods=["POST"])
def add_review():
    if "user_id" not in session:
        flash("You must be logged in to post a review.", "error")
        return redirect(url_for("user_login"))

    review_text = request.form.get("review", "").strip()

    if review_text:
        user = users_col.find_one({"_id": ObjectId(session["user_id"])})

        reviews_col.insert_one({
            "user_id": ObjectId(session["user_id"]),
            "username": user["username"],
            "review": review_text,
            "created_at": datetime.now()
        })
        flash("Review posted successfully!", "success")
    else:
        flash("Review cannot be empty.", "error")

    return redirect(url_for("user_home"))


@app.route("/appointments")
def appointments():
    if "user_id" not in session:
        return redirect(url_for("login"))

    approved_doctors = list(
        doctors_col.find({"approved": True}).sort("created_at", -1)
    )

    # ✅ fetch logged-in user only
    user = users_col.find_one({"_id": ObjectId(session["user_id"])})

    # Convert ObjectId for doctors
    for doc in approved_doctors:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].strftime("%d-%m-%Y")

    return render_template(
        "appointments.html",
        doctors=approved_doctors,
        user=user
    )


@app.route("/recom_appointment")
def recom_appointment():
    specialization = request.args.get("specialization")

    query = {"approved": True}
    if specialization:
        query["specialization"] = {"$regex": specialization, "$options": "i"}  # match partial text

    approved_doctors = list(doctors_col.find(query).sort("created_at", -1))

    user = users_col.find_one({"_id": ObjectId(session["user_id"])})

    for doc in approved_doctors:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].strftime("%d-%m-%Y")

    return render_template("recom_appointment.html", doctors=approved_doctors,user=user)

# ----------------- Book Appointment -----------------
@app.route("/book_appointment", methods=["POST"])
def book_appointment():
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 401

    data = request.json
    doctor_id = data.get("doctor_id")
    user_id = session["user_id"]
    date = data.get("date")
    time = data.get("time")
    phone = data.get("phone")

    if not doctor_id or not date or not time or not phone:
        return jsonify({"error": "Missing fields"}), 400

    # Convert IDs
    try:
        doctor_oid = ObjectId(doctor_id)
        user_oid = ObjectId(user_id)
    except Exception:
        return jsonify({"error": "Invalid IDs"}), 400

    # Check if slot already booked
    existing = appointments_col.find_one({
        "doctor_id": doctor_oid,
        "appointment_date": date,
        "time": time
    })
    if existing:
        return jsonify({"error": "Slot already booked"}), 400
    
    users_col.update_one(
        {"_id": user_oid},
        {"$set": {"phone": phone}}
    )

    # Create appointment
    appointment = {
        "doctor_id": doctor_oid,
        "user_id": user_oid,
        "appointment_date": date,
        "time": time,
        "phone": phone,
        "created_at": datetime.now(),
        "sms_morning_sent": False,
        "sms_1hr_sent": False,
    }
    result = appointments_col.insert_one(appointment)
    
    user = users_col.find_one({"_id": user_oid})
    doctor = doctors_col.find_one({"_id": doctor_oid})

    send_email(
        to=user["email"],
        subject="Appointment Booked – Wellness Wave",
        body=f"""
    Hello {user['username']},

    Your appointment has been booked successfully.

    Doctor: Dr. {doctor['name']}
    Specialization: {doctor['specialization']}
    Date: {date}
    Time: {time}

    Thank you,
    Wellness Wave
    """
    )

    return jsonify({
        "success": True,
        "message": "Appointment booked successfully!",
        "appointment_id": str(result.inserted_id)
    })


# ----------------- Get User Appointments -----------------
@app.route("/my_appointments")
def my_appointments():
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("index"))

    user_id = ObjectId(session["user_id"])
    appointments_list = list(appointments_col.find({"user_id": user_id}).sort("appointment_date", 1))

    # Populate doctor details and convert ObjectIds for Jinja
    for appt in appointments_list:
        doctor = doctors_col.find_one({"_id": ObjectId(appt["doctor_id"])})
        appt["doctor"] = doctor
        appt["_id"] = str(appt["_id"])
        appt["doctor"]["_id"] = str(doctor["_id"]) if doctor else None

    return render_template("my_appointments.html", appointments=appointments_list)


# ----------------- Cancel Appointment -----------------
@app.route("/cancel_appointment/<appt_id>", methods=["POST"])
def cancel_appointment(appt_id):
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 401

    try:
        appt = appointments_col.find_one({"_id": ObjectId(appt_id)})
        if not appt:
            return jsonify({"error": "Appointment not found"}), 404
        if appt["status"] == "cancelled":
            return jsonify({"error": "Appointment already cancelled"}), 400

        appointments_col.update_one(
            {"_id": ObjectId(appt_id)},
            {"$set": {"status": "cancelled"}}
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Page to display past analyses
@app.route("/history")
def history():
    # Check if user is logged in
    if "user_id" not in session:
        return redirect(url_for("login"))  # redirect to login if not logged in

    # Get only this user's analyses
    analyses = list(
        analysis_collection.find({"user_id": session["user_id"]}).sort("datetime", -1)
    )

    return render_template("history.html", analyses=analyses)

#summary
@app.route("/summary/<user_id>")
def summary(user_id):

    # Get latest analysis for THIS user
    latest_analysis = analysis_collection.find_one(
        {"user_id": user_id},
        sort=[("datetime", -1)]
    )

    return render_template(
        "summary.html",
        analysis=latest_analysis
    )


# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))

# major project
# blog page
@app.route("/doctor/<doctor_id>/blogs")
def doctor_blog_page(doctor_id):
    doctor_id = ObjectId(doctor_id)

    blogs = list(
        blog_col.find({"doctor_id": doctor_id}).sort("_id", -1)
    )

    doctor = doctors_col.find_one({"_id": doctor_id})

    return render_template(
        "blog.html",
        blogs=blogs,
        doctor=doctor
    )


#upload blog
@app.route("/upload_blog", methods=["POST"])
def upload_blog():
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    title = request.form.get("blog_title", "").strip()
    content = request.form.get("blog_content", "").strip()

    image = None
    pic = request.files.get("blog_image")

    if pic and pic.filename:
        filename = secure_filename(pic.filename)
        save_path = os.path.join(BLOG_UPLOAD_FOLDER, filename)
        pic.save(save_path)
        image = f"blogs/{filename}"

    upload_date = datetime.now().strftime("%b %d, %Y")

    doctor_id = ObjectId(session["doctor_id"])
    doctor = doctors_col.find_one({"_id": doctor_id})

    blog_col.insert_one({
        "title": title,
        "image": image,
        "content": content,
        "date": upload_date,

        # ✅ DOCTOR INFO
        "doctor_id": doctor_id,
        "doctor_name": doctor["name"],
        "doctor_specialization": doctor.get("specialization", ""),
        "doctor_image": doctor.get("profile_image", "images/default-doctor.png")
    })

    flash("Blog uploaded successfully", "success")

    # ✅ FIXED redirect
    return redirect(url_for("doctor_blog_page", doctor_id=str(doctor_id)))


#remove blog
@app.route("/delete_blog/<blog_id>", methods=["POST"])
def delete_blog(blog_id):
    if "doctor_id" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    doctor_id = ObjectId(session["doctor_id"])
    blog = blog_col.find_one({"_id": ObjectId(blog_id), "doctor_id": doctor_id})

    if not blog:
        return jsonify({"success": False, "message": "Blog not found or not authorized"}), 404

    blog_col.delete_one({"_id": ObjectId(blog_id)})

    return jsonify({"success": True, "message": "Blog deleted"})

# chatbot
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    reply = lifestyle_disease_chat(user_input)
    return jsonify({"reply": reply})

# =========================
# PATIENT CHAT
# =========================
@app.route("/patient/chat/<doctor_id>")
def patient_chat(doctor_id):
    if "user_id" not in session:
        return redirect(url_for("user_login"))

    user_id = session["user_id"]

    # 🔑 SAME ROOM FORMAT FOR BOTH SIDES
    room = f"chat_{user_id}_{doctor_id}"

    # 📩 Fetch messages
    messages = list(
        chat_col.find({"room": room}).sort("timestamp", 1)
    )

    # 🔥 Fetch profiles
    patient = users_col.find_one({"_id": ObjectId(user_id)})
    doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})

    if not doctor:
        return redirect(url_for("user_home"))

    return render_template(
        "patient_chat.html",
        room=room,
        user_id=str(user_id),
        doctor_id=str(doctor_id),
        patient=patient,
        doctor=doctor,
        messages=messages
    )

# =========================
# DOCTOR CHAT
# =========================
@app.route("/doctor/chat/<user_id>")
def doctor_chat(user_id):
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    doctor_id = session["doctor_id"]

    # 🔑 SAME ROOM FORMAT
    room = f"chat_{user_id}_{doctor_id}"

    # 📩 Fetch chat messages
    messages = list(
        chat_col.find({"room": room}).sort("timestamp", 1)
    )

    # 👤 Fetch profiles
    patient = users_col.find_one({"_id": ObjectId(user_id)})
    doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})

    if not patient:
        return redirect(url_for("doctor_home"))

    # 📝 Fetch patient notes (NEW)
    notes = list(
        db.notes.find({"patient_id": ObjectId(user_id)}).sort("created_at", -1)
    )

    return render_template(
        "doctor_chat.html",
        room=room,
        user_id=str(user_id),
        doctor_id=str(doctor_id),
        patient=patient,
        doctor=doctor,
        messages=messages,
        notes=notes   # ✅ send notes to same page
    )


@socketio.on("join")
def join(data):
    if "room" in data:
        join_room(data["room"])


@socketio.on("join_user")
def join_user(data):
    if "user_id" in data:
        join_room(f"user_{str(data['user_id'])}")

@socketio.on("join_doctor")
def join_doctor(data):
    if "doctor_id" in data:
        join_room(f"doctor_{str(data['doctor_id'])}")

@socketio.on("send_message")
def handle_message(data):
    room = data.get("room")
    if not room:
        return

    sender_role = data.get("sender_role")  # "user" or "doctor"
    receiver_id = str(data.get("receiver_id"))

    # Save chat
    chat_col.insert_one({
        "room": room,
        "sender": data.get("sender"),
        "sender_role": sender_role,
        "sender_id": str(data.get("sender_id")),
        "receiver_id": receiver_id,
        "message": data.get("message"),
        "timestamp": datetime.now()
    })

    # Save notification
    notifications_col.insert_one({
        "receiver_id": receiver_id,
        "sender": data.get("sender"),
        "sender_role": sender_role,
        "room": room,
        "read": False,
        "timestamp": datetime.now()
    })

    # 🔔 Decide correct receiver room
    if sender_role == "doctor":
        receiver_room = f"user_{receiver_id}"
    else:
        receiver_room = f"doctor_{receiver_id}"

    # 🔔 Send notification
    emit("new_notification", {
        "sender": data.get("sender"),
        "room": room
    }, room=receiver_room)

    # 💬 Send message
    emit("receive_message", {
        "sender": data.get("sender"),
        "sender_role": sender_role,
        "message": data.get("message")
    }, room=room)


#file upload
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return {"error": "No file"}, 400

    if not allowed_file(file.filename):
        return {"error": "File type not allowed"}, 400

    filename = secure_filename(
        f"{int(datetime.utcnow().timestamp())}_{file.filename}"
    )

    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    return {"file_url": f"/static/uploads/{filename}"}

#notification
@app.route("/notifications/patient")
def patient_notifications():
    if "user_id" not in session:
        return jsonify([])

    notifs = notifications_col.find({
        "receiver_id": str(session["user_id"]),
        "read": False
    })

    result = []
    for n in notifs:
        result.append({
            "sender": n.get("sender", "Doctor"),  # ✅ SAFE
            "room": n.get("room", "")
        })

    return jsonify(result)

@app.route("/notifications/doctor")
def doctor_notifications():
    if "doctor_id" not in session:
        return jsonify([])

    notifs = notifications_col.find({
        "receiver_id": str(session["doctor_id"]),
        "read": False
    })

    result = []
    for n in notifs:
        result.append({
            "sender": n.get("sender", "Patient"),
            "room": n.get("room", "")
        })

    return jsonify(result)


@app.route("/notification/read/<room>")
def mark_read(room):
    # 1️⃣ Mark notifications as read
    notifications_col.update_many(
        {"room": room},
        {"$set": {"read": True}}
    )

    # 2️⃣ Validate room format: chat_userId_doctorId
    parts = room.split("_")
    if len(parts) != 3:
        if "doctor_id" in session:
            return redirect(url_for("doctor_home"))
        return redirect(url_for("user_home"))

    user_id = parts[1]
    doctor_id = parts[2]

    # 3️⃣ Doctor clicked (priority)
    if "doctor_id" in session:
        return redirect(url_for("doctor_chat", user_id=user_id))

    # 4️⃣ User clicked
    if "user_id" in session:
        return redirect(url_for("patient_chat", doctor_id=doctor_id))

    # 5️⃣ Fallback
    return redirect(url_for("user_login"))

#all blogs
@app.route("/all_blog")
def all_blog():
    latest_blogs = list(
        blog_col.find().sort("_id", -1)
    )
    return render_template("all_blog.html",latest_blogs=latest_blogs)

#save note
@app.route("/add-note", methods=["POST"])
def add_note():
    if "doctor_id" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json

    try:
        patient_id = ObjectId(data.get("patient_id"))
        doctor_id = ObjectId(session["doctor_id"])
    except:
        return jsonify({"status": "error", "message": "Invalid IDs"}), 400

    notes_text = data.get("notes", "").strip()
    if not notes_text:
        return jsonify({"status": "error", "message": "Empty note"}), 400

    notes_list = [
        n.strip("• ").strip()
        for n in notes_text.split("\n")
        if n.strip()
    ]

    db.notes.insert_one({
        "doctor_id": doctor_id,
        "patient_id": patient_id,
        "notes": notes_list,
        "created_at": datetime.utcnow()
    })

    return jsonify({"status": "ok"})

@app.route("/get-notes/<patient_id>")
def get_notes(patient_id):
    if "doctor_id" not in session:
        return jsonify([])

    try:
        notes = list(db.notes.find({
            "patient_id": ObjectId(patient_id)
        }).sort("created_at", -1))

        result = []
        for n in notes:
            result.append({
                "_id": str(n["_id"]),
                "notes": n.get("notes", []),
                "created_at": n.get("created_at", "").strftime("%d %b %Y")
                if n.get("created_at") else ""
            })

        return jsonify(result)

    except Exception as e:
        print("GET NOTES ERROR:", e)
        return jsonify([])

def send_email(to, subject, body):
    try:
        msg = Message(subject, recipients=[to])
        msg.body = body
        mail.send(msg)
        print("Email sent to", to)
    except Exception as e:
        print("EMAIL ERROR:", e)

def appointment_email_reminder():
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)

    appointments = appointments_col.find({
        "status": "pending"
    })

    for appt in appointments:
        try:
            appt_date = datetime.strptime(appt["appointment_date"], "%Y-%m-%d")
            time_str, period = appt["time"].split()
            hour, minute = map(int, time_str.split(":"))

            if period == "PM" and hour != 12:
                hour += 12
            if period == "AM" and hour == 12:
                hour = 0

            appt_datetime = tz.localize(
                datetime(
                    appt_date.year,
                    appt_date.month,
                    appt_date.day,
                    hour,
                    minute
                )
            )

            user = users_col.find_one({"_id": appt["user_id"]})
            doctor = doctors_col.find_one({"_id": appt["doctor_id"]})

            # 🌅 MORNING REMINDER (8 AM)
            morning_time = appt_datetime.replace(hour=8, minute=0)

            if (
                now >= morning_time
                and not appt.get("sms_morning_sent")
            ):
                send_email(
                    to=user["email"],
                    subject="Appointment Reminder – Today",
                    body=f"""
Good Morning {user['username']},

This is a reminder that you have an appointment TODAY.

Doctor: Dr. {doctor['name']}
Time: {appt['time']}

Please be available.

– Wellness Wave
"""
                )
                appointments_col.update_one(
                    {"_id": appt["_id"]},
                    {"$set": {"sms_morning_sent": True}}
                )

            # ⏰ 1-HOUR REMINDER
            one_hour_before = appt_datetime - timedelta(hours=1)

            if (
                now >= one_hour_before
                and not appt.get("sms_1hr_sent")
            ):
                send_email(
                    to=user["email"],
                    subject="Appointment in 1 Hour",
                    body=f"""
Hello {user['username']},

Your appointment is in 1 hour.

Doctor: Dr. {doctor['name']}
Time: {appt['time']}

Please be ready.

– Wellness Wave
"""
                )
                appointments_col.update_one(
                    {"_id": appt["_id"]},
                    {"$set": {"sms_1hr_sent": True}}
                )

        except Exception as e:
            print("REMINDER ERROR:", e)
    
#  Initialize scheduler
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.add_job(appointment_email_reminder, "interval", minutes=1)
scheduler.start()


if __name__ == "__main__":
    socketio.run(app,debug=True)