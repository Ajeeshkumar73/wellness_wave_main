from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import pandas as pd
import joblib
from flask_socketio import SocketIO, emit, join_room
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import pytz



from bot import lifestyle_disease_chat

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
    MAIL_USERNAME="ajeeshexatech@gmail.com",
    MAIL_PASSWORD="tttr ntvl prlq foqk",
    MAIL_DEFAULT_SENDER="Wellness Wave <ajeeshexatech@gmail.com>"
)

mail = Mail(app)

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


# Load all models
diseases = ["Obesity", "Hypertension", "Diabetes", "HeartDisease"]
models = {d: joblib.load(f"{d}_model.pkl") for d in diseases}

def risk_status(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Intermediate Risk"
    else:
        return "Low Risk"

def metric_status(name, value):
    if name == "BMI":
        if value < 18.5: return "Underweight"
        elif value < 25: return "Normal"
        elif value < 30: return "Overweight"
        else: return "Obese"
    if name == "BloodSugar":
        if value < 100: return "Normal"
        elif value < 126: return "Prediabetes"
        else: return "High"
    if name == "BloodPressure":
        s,d = value
        if s < 120 and d < 80: return "Normal"
        elif s < 130 and d < 80: return "Elevated"
        elif s < 140 or d < 90: return "High BP Stage 1"
        else: return "High BP Stage 2"
    if name == "Cholesterol":
        if value < 200: return "Desirable"
        elif value < 240: return "Borderline High"
        else: return "High"
    return "Unknown"

# Full precautions dictionary (converted from your JS object)
precautions_dict = {
    "BMI": {
        "Underweight": {
            "diet": [
                "Increase protein intake (eggs, lean meat, legumes) to support muscle growth and repair",
                "Eat calorie-dense healthy foods like nuts, seeds, avocado, and whole grains to gain weight safely",
                "Have 5-6 small meals daily instead of skipping meals to maintain steady energy and nutrient supply",
                "Include healthy snacks between meals such as yogurt, smoothies, or nut butter on whole-grain toast"
            ],
            "exercise": [
                "Engage in light resistance training to build lean muscle mass without excessive strain",
                "Avoid excessive cardio which can burn too many calories and hinder weight gain",
                "Focus on exercises like squats, push-ups, and resistance band workouts that are manageable at your level"
            ],
            "habit": [
                "Maintain consistent meal timing every day to regulate metabolism",
                "Get adequate sleep (7-9 hours) to support recovery and healthy weight gain",
                "Track weight weekly and adjust diet/exercise based on progress",
                "Reduce stress with relaxation techniques such as deep breathing or meditation"
            ]
        },
        "Overweight": {
            "diet": [
                "Reduce sugar and refined carbs like white bread, pastries, and sugary drinks to avoid excess fat storage",
                "Eat more vegetables, fiber, and lean proteins to improve satiety and nutrient intake",
                "Control portion sizes using smaller plates and mindful eating practices",
                "Avoid skipping meals which can lead to overeating later in the day"
            ],
            "exercise": [
                "Engage in 30-45 min of cardio 5 days/week (walking, jogging, cycling) to burn extra calories",
                "Include strength training 2-3 times/week to build muscle and boost metabolism",
                "Try interval training for higher calorie burn and cardiovascular benefit"
            ],
            "habit": [
                "Avoid late-night snacking, especially high-calorie foods",
                "Track food intake and weight regularly to monitor progress",
                "Manage stress through meditation, yoga, or breathing exercises",
                "Maintain consistent sleep schedule to regulate appetite hormones"
            ]
        },
        "Obese": {
            "diet": [
                "Consult a nutritionist for a personalized calorie-controlled diet plan to ensure safe weight loss",
                "Follow a calorie deficit meal plan while ensuring adequate nutrition",
                "Limit processed foods, sugary drinks, and high-fat items",
                "Incorporate plenty of fiber-rich foods like vegetables, fruits, and whole grains to feel full"
            ],
            "exercise": [
                "Participate in a supervised daily exercise program suited to your fitness level",
                "Include a mix of cardio (walking, swimming, cycling) and strength training",
                "Start with shorter sessions and gradually increase intensity to avoid injury"
            ],
            "habit": [
                "Attend regular medical checkups to monitor health markers",
                "Set realistic goals and track progress using apps or journals",
                "Focus on improving sleep quality and stress management",
                "Stay motivated by setting achievable milestones and rewarding progress"
            ]
        }
    },
    "BloodPressure": {
        "Elevated": {
            "diet": [
                "Reduce sodium intake by avoiding processed foods, canned soups, and salty snacks",
                "Limit sugary drinks and high-calorie beverages to support weight and blood pressure control",
                "Increase potassium-rich foods like bananas, spinach, and sweet potatoes to help balance sodium levels",
                "Eat a variety of fresh fruits, vegetables, lean proteins, and whole grains daily"
            ],
            "exercise": [
                "Walk briskly for at least 30 min every day to improve cardiovascular health",
                "Include light strength exercises to enhance overall fitness",
                "Gradually increase intensity as endurance improves while monitoring heart rate"
            ],
            "habit": [
                "Practice stress management techniques such as yoga, meditation, or deep breathing",
                "Avoid smoking and limit alcohol consumption to protect heart and blood vessels",
                "Monitor blood pressure regularly and keep a log for your doctor",
                "Maintain healthy sleep patterns for overall cardiovascular support"
            ]
        },
        "High Stage 1": {
            "diet": [
                "Adopt a low-salt, low-fat diet with emphasis on fresh vegetables and lean proteins",
                "Avoid processed and fried foods that can raise blood pressure",
                "Include whole grains and fiber-rich foods to support heart health",
                "Limit red meat and sugary foods, and drink plenty of water"
            ],
            "exercise": [
                "Perform cardio exercises 30-45 min, 5 days/week",
                "Incorporate strength training 2-3 times/week to maintain muscle mass",
                "Include flexibility exercises like stretching or yoga to support circulation"
            ],
            "habit": [
                "Monitor blood pressure regularly and note changes",
                "Limit alcohol intake and avoid smoking",
                "Manage stress daily using meditation, deep breathing, or mindfulness",
                "Get regular checkups to prevent progression to higher stages"
            ]
        },
        "High Stage 2": {
            "diet": [
                "Consult a dietitian for a personalized low-salt, heart-healthy diet plan",
                "Avoid processed, fried, and high-fat foods",
                "Eat more vegetables, fruits, whole grains, and lean proteins",
                "Follow meal timings and avoid skipping meals to maintain stable blood pressure"
            ],
            "exercise": [
                "Engage in a doctor-supervised exercise program",
                "Include gentle cardio and resistance training suited to your condition",
                "Start slow and increase intensity gradually under supervision"
            ],
            "habit": [
                "Follow prescribed medication regimen strictly",
                "Monitor blood pressure at home daily and log readings",
                "Incorporate stress management practices every day",
                "Attend regular medical appointments for evaluation"
            ]
        }
    },
    "Cholesterol": {
        "Borderline High": {
            "diet": [
                "Reduce saturated fats found in butter, cheese, and fatty meats",
                "Eat more fiber-rich foods like oats, beans, and fruits to lower LDL cholesterol",
                "Include fatty fish like salmon or mackerel twice a week for omega-3 benefits",
                "Avoid trans fats found in fried foods and baked goods"
            ],
            "exercise": [
                "Engage in brisk walking or light jogging 30 min daily",
                "Include moderate cardio and light strength exercises 2-3 times/week",
                "Increase physical activity gradually to improve lipid profile"
            ],
            "habit": [
                "Quit smoking to improve heart health",
                "Limit alcohol intake to recommended guidelines",
                "Monitor cholesterol levels periodically",
                "Maintain healthy weight and sleep routines"
            ]
        },
        "High": {
            "diet": [
                "Consult a dietitian for a heart-healthy meal plan",
                "Strictly limit fried, processed, and high-fat foods",
                "Increase intake of fruits, vegetables, and whole grains",
                "Focus on lean protein sources like chicken, fish, and legumes"
            ],
            "exercise": [
                "Follow a supervised exercise plan with cardio and strength training 4-5 times/week",
                "Engage in at least 150 min of moderate-intensity exercise weekly",
                "Include flexibility and stretching exercises to reduce injury risk"
            ],
            "habit": [
                "Take medications as prescribed by your doctor",
                "Monitor cholesterol regularly with lab tests",
                "Incorporate stress reduction techniques such as meditation",
                "Avoid smoking and excessive alcohol consumption"
            ]
        }
    },
    "BloodSugar": {
        "Prediabetes": {
            "diet": [
                "Reduce sugar, sugary drinks, and refined carbohydrates like white bread and pastries",
                "Increase fiber intake through vegetables, legumes, and whole grains",
                "Eat smaller, frequent meals to maintain stable blood glucose",
                "Avoid skipping meals to prevent spikes in blood sugar"
            ],
            "exercise": [
                "Do 30 min of moderate exercise daily like brisk walking or cycling",
                "Include strength training 2-3 times/week to improve insulin sensitivity",
                "Incorporate light stretching or yoga to enhance circulation"
            ],
            "habit": [
                "Monitor blood sugar regularly",
                "Maintain healthy weight through diet and exercise",
                "Manage stress with mindfulness or relaxation techniques",
                "Ensure adequate sleep every night (7-9 hours)"
            ]
        },
        "High": {
            "diet": [
                "Consult a dietitian for a personalized low-sugar, high-fiber diet",
                "Strictly limit sugar, processed foods, and refined carbs",
                "Include plenty of vegetables, whole grains, and lean protein",
                "Follow consistent meal timings and avoid late-night eating"
            ],
            "exercise": [
                "Follow a supervised exercise program with a mix of cardio and resistance training",
                "Gradually increase exercise duration and intensity under supervision",
                "Include daily walking or swimming to control blood sugar levels"
            ],
            "habit": [
                "Take prescribed medications as directed",
                "Monitor glucose regularly and log readings",
                "Practice stress management and ensure good sleep hygiene",
                "Attend regular medical checkups to adjust treatment if needed"
            ]
        }
    },
    "Diabetes": {
        "diet": ["Low glycemic index diet", "Eat more vegetables, whole grains, and lean protein", "Avoid sugary beverages"],
        "exercise": ["30-60 min daily walking or cardio", "Strength/resistance exercises 2-3 times/week"],
        "habit": ["Monitor blood glucose daily", "Take medications as prescribed", "Maintain healthy weight and stress control"]
    },
    "Heart Disease": {
        "diet": ["Heart-healthy diet with fruits, vegetables, and whole grains", "Limit saturated fats and trans fats", "Include omega-3 rich foods like fish or flaxseeds"],
        "exercise": ["Moderate-intensity cardio 150 min/week", "Strength training 2-3 times/week"],
        "habit": ["Avoid smoking and excessive alcohol", "Regular checkups and ECG monitoring", "Stress management techniques"]
    },
    "Hypertension": {
        "diet": ["Low-salt diet", "Eat potassium-rich foods", "Avoid processed and fried foods"],
        "exercise": ["30-45 min daily moderate exercise", "Strength training 2x/week"],
        "habit": ["Monitor blood pressure regularly", "Stress management", "Avoid alcohol and smoking"]
    },
    "Obesity": {
        "diet": ["Calorie deficit diet with balanced nutrition", "Avoid sugary and processed foods", "Increase intake of vegetables, lean protein, and fiber"],
        "exercise": ["Daily physical activity (cardio + strength)", "Gradually increase intensity"],
        "habit": ["Track weight and meals", "Maintain good sleep hygiene", "Manage stress levels"]
    }
}


# Function to compute precautions
def compute_precautions(metrics, disease_results):
    """
    metrics: dict of metric -> {'status': str}, e.g.,
        {"BMI": {"status": "Overweight"}, "BloodPressure": {"status": "High Stage 1"}}
    disease_results: dict of disease -> {'status': str}, e.g.,
        {"Diabetes": {"status": "High Risk"}}
    """
    result_precautions = {"diet": [], "exercise": [], "habit": []}

    # Metric-based precautions
    for metric, val in metrics.items():
        status = val['status']
        if metric in precautions_dict:
            metric_prec = precautions_dict[metric]
            # Some metrics like BMI or BloodPressure have nested statuses
            if isinstance(metric_prec, dict) and status in metric_prec:
                for key in ["diet", "exercise", "habit"]:
                    result_precautions[key] += metric_prec[status].get(key, [])
            # Some metrics like Diabetes have direct lists
            elif isinstance(metric_prec, dict):
                for key in ["diet", "exercise", "habit"]:
                    if key in metric_prec:
                        result_precautions[key] += metric_prec[key]

    # Disease-based precautions
    for disease, info in disease_results.items():
        status = info['status']
        if disease in precautions_dict:
            disease_prec = precautions_dict[disease]
            # Only apply if high risk or high status
            if status in ["High Risk", "High"]:
                for key in ["diet", "exercise", "habit"]:
                    if key in disease_prec:
                        result_precautions[key] += disease_prec[key]

    # Remove duplicates
    for key in result_precautions:
        result_precautions[key] = list(dict.fromkeys(result_precautions[key]))

    return result_precautions

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    height = float(data['height'])
    weight = float(data['weight'])
    bmi = weight / ((height / 100) ** 2)
    
    input_data = pd.DataFrame([{
        "Age": int(data['age']),
        "Gender": data['gender'],
        "Height_cm": height,
        "Weight_kg": weight,
        "Smoking": data['smoking'],
        "Alcohol": data['alcohol'],
        "Exercise_Freq": data['exercise'],
        "Sleep_Hours": float(data['sleep']),
        "SystolicBP": float(data['systolic']),
        "DiastolicBP": float(data['diastolic']),
        "Cholesterol": float(data['cholesterol']),
        "BloodSugar": float(data['bloodSugar']),
        "BMI": bmi
    }])

    input_data = pd.get_dummies(input_data)
    

    results = {}

    # Run predictions for each model
    for disease in diseases:
        model = models[disease]
        # align features for this model
        model_input = input_data.copy()
        for col in model.feature_names_in_:
            if col not in model_input.columns:
                model_input[col] = 0
        model_input = model_input[model.feature_names_in_]

        prob = model.predict_proba(model_input)[0][1]
        status = risk_status(prob)

        results[disease] = {
            "chance": f"{prob*100:.1f}%",
            "status": status
        }



    metrics = {
        "BMI": {"value": round(bmi,1), "status": metric_status("BMI", bmi)},
        "BloodPressure": {"value": f"{data['systolic']}/{data['diastolic']}", 
                          "status": metric_status("BloodPressure", (float(data['systolic']), float(data['diastolic'])))},
        "BloodSugar": {"value": data['bloodSugar'], "status": metric_status("BloodSugar", float(data['bloodSugar']))},
        "Cholesterol": {"value": data['cholesterol'], "status": metric_status("Cholesterol", float(data['cholesterol']))},
    }

     # Compute precautions
    precautions = compute_precautions(metrics, results)

    # Save analysis to MongoDB
    analysis_collection.insert_one({
         "user_id": session["user_id"], 
        "metrics": metrics,
        "diseases": results,
        "precautions": precautions if precautions else {"diet": [], "exercise": [], "habit": []},
        "datetime": datetime.now()
    })
                
    return jsonify({"diseases": results, "metrics": metrics, "precautions": precautions})




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

    return render_template("admin_page.html", users=users, doctors=doctors)


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

              # Store only relative path for DB
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

        flash("Doctor registered successfully! Waiting for admin approval.", "success")
        return redirect(url_for("index", open_login="true"))

    # If GET → render the registration form
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
    if "username" not in session:
        flash("You must be logged in to post a review.", "error")
        return redirect(url_for("user_login"))

    review_text = request.form.get("review", "").strip()

    if review_text:
        reviews_col.insert_one({
            "username": session["username"],
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

    for doc in approved_doctors:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].strftime("%d-%m-%Y")

    return render_template("recom_appointment.html", doctors=approved_doctors)

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
    
# 3️⃣ Initialize scheduler
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.add_job(appointment_email_reminder, "interval", minutes=1)
scheduler.start()


if __name__ == "__main__":
    socketio.run(app,debug=True)