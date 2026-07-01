# 🌊 WellnessWave — AI-Powered Health Risk Assessment Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.1-black?logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-4.x-green?logo=mongodb)
![PyTorch](https://img.shields.io/badge/PyTorch-TabNet-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)

WellnessWave is a full-stack web application that uses **deep learning (TabNet)** to predict the risk percentage of lifestyle diseases — **Diabetes, Heart Disease, Hypertension, and Obesity** — based on user health inputs. It also features real-time doctor–patient chat, appointment booking, AI-generated health precautions via Groq LLM, a health blog, and downloadable PDF reports.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **AI Risk Prediction** | TabNet deep learning models predict risk % for 4 lifestyle diseases |
| 💡 **Model Explainability** | Top contributing health factors shown per disease |
| 🤖 **AI Health Chatbot** | Groq-powered LLM (Qwen) answers health-related questions in context |
| 📋 **AI Precautions** | Personalized diet, exercise, and habit recommendations via Groq API |
| 📄 **PDF Report Download** | Professional health report generated with ReportLab |
| 👨‍⚕️ **Doctor Portal** | Doctors can manage appointments, respond to chats, and post blogs |
| 🗓️ **Appointment Booking** | Patients can book, track, and manage appointments |
| 💬 **Real-Time Chat** | WebSocket-based doctor–patient messaging (Flask-SocketIO) |
| 📧 **Email Notifications** | Appointment reminders sent via Flask-Mail (Gmail SMTP) |
| 📝 **Health Blog** | Doctors publish articles; users can read and interact |
| 📊 **History Dashboard** | View past health analysis results |
| 🔐 **Auth System** | Secure user and doctor registration/login with hashed passwords |
| 🔑 **Password Reset** | Token-based forgot password via email |
| 🛡️ **Admin Panel** | Admin can manage users, doctors, and platform content |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask 3.1.1, Python 3.10+ |
| **Database** | MongoDB (via PyMongo) |
| **ML Models** | PyTorch-TabNet (TabNetRegressor) |
| **AI / LLM** | Groq API (Qwen 3 32B model) |
| **Real-Time** | Flask-SocketIO (WebSockets) |
| **Email** | Flask-Mail (Gmail SMTP) |
| **PDF** | ReportLab |
| **Scheduling** | APScheduler (appointment reminders) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, Jinja2 templating |

---

## 📁 Project Structure

```
wellnesswave-main/
│
├── app.py                        # Main Flask application (routes, logic)
├── bot.py                        # Groq AI health chatbot module
├── precaution.py                 # Groq AI precaution generator
├── train_model.py                # TabNet model training script
├── check.py                      # Utility/debug script
│
├── *.zip                         # Pre-trained TabNet models (4 diseases)
├── label_encoders.pkl            # Saved LabelEncoders for categorical features
├── scaler.pkl                    # Saved StandardScaler
├── health_risk_dataset.csv       # Training dataset
│
├── templates/                    # Jinja2 HTML templates
│   ├── index.html                # Landing page
│   ├── users_login.html
│   ├── users_register.html
│   ├── user_home.html
│   ├── predict_page.html         # Health input form
│   ├── summary.html              # Prediction results & explainability
│   ├── history.html              # Past analysis history
│   ├── doctor_home.html
│   ├── doctor_login.html
│   ├── doctor_register.html
│   ├── doctor_chat.html
│   ├── patient_chat.html
│   ├── appointments.html
│   ├── my_appointments.html
│   ├── manage_appointments.html
│   ├── Recom_appointment.html
│   ├── blog.html
│   ├── all_blog.html
│   ├── admin_page.html
│   ├── forgot_password.html
│   └── reset_password.html
│
├── static/                       # CSS, JS, images, uploads
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not committed)
└── .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- MongoDB running locally on `mongodb://localhost:27017/`
- A [Groq API Key](https://console.groq.com/)
- A Gmail account for email notifications (with App Password enabled)

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/wellnesswave.git
cd wellnesswave
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`):

```env
SESSION_SECRET=your_flask_secret_key_here
MAIL_PASSWORD=your_gmail_app_password_here
GROQ_API_KEY=your_groq_api_key_here
```

> **How to get a Gmail App Password:**
> Go to Google Account → Security → 2-Step Verification → App Passwords → Generate one for "Mail".

### 5. Start MongoDB

Ensure MongoDB is running locally:

```bash
# Windows (if installed as a service)
net start MongoDB

# Or start manually
mongod
```

### 6. Run the Application

```bash
python app.py
```

Visit **http://localhost:5000** in your browser.

---

## 🤖 Machine Learning Models

The app uses **four TabNet models** (one per disease), pre-trained on a health risk dataset with **27 input features**:

| Feature | Feature | Feature |
|---|---|---|
| Age | Gender | Height (cm) |
| Weight (kg) | BMI | Body Fat % |
| Family History: Diabetes | Family History: Heart Disease | Existing BP Issues |
| Cholesterol (mg/dL) | Physical Activity (days/wk) | Smoking |
| Sleep Hours | Sleep Quality | Junk Food/wk |
| Salt Intake | Systolic BP | Diastolic BP |
| Shortness of Breath | Waist Circumference (cm) | Frequent Urination |
| Excessive Thirst | Red Meat/wk | Fried Food/wk |
| Water Intake (L/day) | Alcohol Consumption | Blood Sugar (mg/dL) |

To retrain the models on new data, run:

```bash
python train_model.py
```

---

## 🔑 Environment Variables Reference

| Variable | Description | Required |
|---|---|---|
| `SESSION_SECRET` | Flask session secret key | ✅ Yes |
| `MAIL_PASSWORD` | Gmail App Password for SMTP | ✅ Yes |
| `GROQ_API_KEY` | Groq API key for LLM features | ✅ Yes |

---

## 👥 User Roles

| Role | Capabilities |
|---|---|
| **Patient/User** | Register, login, run health predictions, book appointments, chat with doctors, view history |
| **Doctor** | Register (pending admin approval), manage appointments, chat with patients, post blogs |
| **Admin** | Manage users and doctors, view all data, moderate content |

---

## 📧 Email Features

- **Appointment Confirmation** — Sent to patient on booking
- **Appointment Reminders** — Scheduled via APScheduler (24h before)
- **Password Reset** — Token-based secure link sent via email

---

## 🚀 Deployment Notes

- Set `debug=False` in `app.py` before deploying to production
- Use a production WSGI server like **Gunicorn** with **eventlet**:
  ```bash
  gunicorn --worker-class eventlet -w 1 app:app
  ```
- Use **MongoDB Atlas** for cloud database hosting
- Store secrets in environment variables or a secrets manager — never commit `.env`

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [PyTorch-TabNet](https://github.com/dreamquark-ai/tabnet) — Tabular deep learning model
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [Flask](https://flask.palletsprojects.com/) — Lightweight Python web framework
- [ReportLab](https://www.reportlab.com/) — PDF generation
- [MongoDB](https://www.mongodb.com/) — NoSQL database
