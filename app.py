"""
Smart Crop Recommendation System
Flask + Random Forest

Run:  python app.py   ->  http://127.0.0.1:5001
History page          ->  http://127.0.0.1:5001/history
Model performance     ->  http://127.0.0.1:5001/performance
Dashboard             ->  http://127.0.0.1:5001/dashboard
Report                ->  http://127.0.0.1:5001/report
Login / Register      ->  http://127.0.0.1:5001/login
Forgot password       ->  http://127.0.0.1:5001/forgot
"""
import os
import io
import csv
import smtplib
import json
import sqlite3
import joblib
from datetime import datetime
from email.message import EmailMessage
from functools import wraps
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
from flask import (Flask, Response, g, jsonify, redirect, render_template,
                   request, session, url_for)
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
DB_PATH = os.path.join(BASE_DIR, "crop_app.db")

app = Flask(__name__)
# Change this secret key before deploying. It signs the login session cookie.
app.secret_key = os.environ.get("SECRET_KEY", "smart-crop-change-this-secret-key")


# --------------------------------------------------------------------------
# Model loading (works if the .pkl is in model/ or next to app.py)
# --------------------------------------------------------------------------
def load_model():
    candidates = [
        os.path.join(BASE_DIR, "model", "crop_model.pkl"),
        os.path.join(BASE_DIR, "crop_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(
        "crop_model.pkl not found. Run train_model.py first so the model file is created."
    )


model = load_model()

# --------------------------------------------------------------------------
# Input fields (slider range follows the real dataset; "hard" only blocks nonsense)
# --------------------------------------------------------------------------
FIELDS = [
    {"key": "N", "label": "Nitrogen (N)", "unit": "kg/ha", "min": 0, "max": 140, "step": 1, "default": 50, "hard": (0, 300)},
    {"key": "P", "label": "Phosphorus (P)", "unit": "kg/ha", "min": 5, "max": 145, "step": 1, "default": 50, "hard": (0, 300)},
    {"key": "K", "label": "Potassium (K)", "unit": "kg/ha", "min": 5, "max": 205, "step": 1, "default": 50, "hard": (0, 300)},
    {"key": "temperature", "label": "Temperature", "unit": "°C", "min": 8, "max": 44, "step": 0.1, "default": 25, "hard": (-10, 60)},
    {"key": "humidity", "label": "Humidity", "unit": "%", "min": 14, "max": 100, "step": 0.1, "default": 70, "hard": (0, 100)},
    {"key": "ph", "label": "Soil pH", "unit": "pH", "min": 3.5, "max": 9.9, "step": 0.1, "default": 6.5, "hard": (0, 14)},
    {"key": "rainfall", "label": "Rainfall", "unit": "mm", "min": 20, "max": 300, "step": 1, "default": 100, "hard": (0, 1000)},
]

# Quick sample values so the project can be demonstrated in one click
PRESETS = [
    {"name": "Rice field", "emoji": "🌾", "data": {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82, "ph": 6.5, "rainfall": 202}},
    {"name": "Cotton farm", "emoji": "☁️", "data": {"N": 120, "P": 40, "K": 20, "temperature": 24, "humidity": 80, "ph": 6.9, "rainfall": 80}},
    {"name": "Apple orchard", "emoji": "🍎", "data": {"N": 20, "P": 134, "K": 200, "temperature": 22.6, "humidity": 92, "ph": 5.9, "rainfall": 112}},
    {"name": "Coffee estate", "emoji": "☕", "data": {"N": 100, "P": 28, "K": 30, "temperature": 25, "humidity": 58, "ph": 6.8, "rainfall": 158}},
]

# --------------------------------------------------------------------------
# Crop knowledge base (all 22 crops of the dataset)
# --------------------------------------------------------------------------
CROPS = {
    "rice": {"emoji": "🌾", "season": "Kharif (Jun - Nov)", "soil": "Clayey soil", "fertilizer": "Urea",
             "description": "Rice is a cereal crop that grows well in warm, humid weather and needs standing water during growth."},
    "maize": {"emoji": "🌽", "season": "Kharif (Jun - Oct)", "soil": "Loamy soil", "fertilizer": "NPK",
              "description": "Maize is grown for food, animal feed and industry. It prefers well-drained soil and moderate rain."},
    "chickpea": {"emoji": "🫘", "season": "Rabi (Oct - Mar)", "soil": "Sandy loam soil", "fertilizer": "DAP",
                 "description": "Chickpea is a protein-rich pulse that grows in cool, dry weather and needs little water."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif (Jun - Sep)", "soil": "Loamy soil", "fertilizer": "Compost",
                    "description": "Kidney beans like mild temperature and well-drained, fertile soil rich in organic matter."},
    "pigeonpeas": {"emoji": "🌱", "season": "Kharif (Jun - Dec)", "soil": "Well-drained loamy soil", "fertilizer": "DAP",
                   "description": "Pigeon pea (tur / arhar) is a hardy pulse that tolerates dry spells and improves soil nitrogen."},
    "mothbeans": {"emoji": "🌱", "season": "Kharif (Jul - Oct)", "soil": "Sandy soil", "fertilizer": "Compost",
                  "description": "Moth bean is a drought-resistant pulse that grows well in hot, dry regions."},
    "mungbean": {"emoji": "🌱", "season": "Zaid / Kharif", "soil": "Loamy soil", "fertilizer": "Organic compost",
                 "description": "Mung bean (green gram) is a short-duration pulse that matures in about 60 to 70 days."},
    "blackgram": {"emoji": "🫘", "season": "Kharif (Jun - Sep)", "soil": "Clay loam soil", "fertilizer": "NPK",
                  "description": "Black gram (urad) is a pulse that needs warm, humid weather and well-drained soil."},
    "lentil": {"emoji": "🫘", "season": "Rabi (Oct - Mar)", "soil": "Loamy soil", "fertilizer": "DAP",
               "description": "Lentil is a cool-season pulse, rich in protein, that grows on light to medium soil."},
    "pomegranate": {"emoji": "🍎", "season": "All season (fruit in Sep - Feb)", "soil": "Sandy loam soil", "fertilizer": "Organic manure",
                    "description": "Pomegranate is a fruit crop that handles heat and dry weather and gives fruit for many years."},
    "banana": {"emoji": "🍌", "season": "All season", "soil": "Rich loamy soil", "fertilizer": "Potash",
               "description": "Banana is a tropical fruit crop that needs high humidity, regular water and potassium-rich soil."},
    "mango": {"emoji": "🥭", "season": "Summer (fruit in Apr - Jul)", "soil": "Deep loamy soil", "fertilizer": "Organic compost",
              "description": "Mango is the king of fruits. It needs a dry spell before flowering and warm weather for fruit."},
    "grapes": {"emoji": "🍇", "season": "Winter pruning, summer harvest", "soil": "Sandy loam soil", "fertilizer": "NPK",
               "description": "Grapes grow on trellises and need warm, dry weather with well-drained soil."},
    "watermelon": {"emoji": "🍉", "season": "Zaid (Feb - May)", "soil": "Sandy loam soil", "fertilizer": "Potash",
                   "description": "Watermelon is a summer fruit that loves heat and sandy, well-drained soil."},
    "muskmelon": {"emoji": "🍈", "season": "Zaid (Feb - May)", "soil": "Sandy loam soil", "fertilizer": "Compost",
                  "description": "Muskmelon is a sweet summer fruit that grows best in warm, dry weather."},
    "apple": {"emoji": "🍎", "season": "Winter (fruit in Aug - Nov)", "soil": "Loamy soil", "fertilizer": "Organic fertilizer",
              "description": "Apple is a temperate fruit crop that needs a cold winter and cool hill climate."},
    "orange": {"emoji": "🍊", "season": "Winter (fruit in Nov - Feb)", "soil": "Sandy loam soil", "fertilizer": "NPK",
               "description": "Orange is a citrus fruit that grows in sub-tropical weather with good drainage."},
    "papaya": {"emoji": "🥭", "season": "All season", "soil": "Well-drained loamy soil", "fertilizer": "Urea",
               "description": "Papaya is a fast-growing tropical fruit that gives fruit within a year but cannot stand waterlogging."},
    "coconut": {"emoji": "🥥", "season": "All season", "soil": "Sandy / coastal soil", "fertilizer": "Potassium sulphate",
                "description": "Coconut is a coastal plantation crop that needs high humidity and plenty of rain."},
    "cotton": {"emoji": "☁️", "season": "Kharif (Apr - Oct)", "soil": "Black soil", "fertilizer": "DAP",
               "description": "Cotton is an important fibre crop for the textile industry and grows best in black soil."},
    "jute": {"emoji": "🌿", "season": "Kharif (Mar - Jul)", "soil": "Alluvial soil", "fertilizer": "Urea",
             "description": "Jute is a natural fibre crop (the golden fibre) that needs hot, humid weather and heavy rain."},
    "coffee": {"emoji": "☕", "season": "Winter harvest (Nov - Feb)", "soil": "Red / loamy soil", "fertilizer": "Organic compost",
               "description": "Coffee is a plantation crop grown on hill slopes with shade, moderate rain and mild temperature."},
}

DEFAULT_INFO = {"emoji": "🌱", "season": "Not available", "soil": "Not available",
                "fertilizer": "General purpose fertilizer", "description": "Information not available."}

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "svg")


def crop_image_url(crop):
    """Return the URL of static/images/<crop>.<ext> or None if the photo is missing."""
    for ext in IMAGE_EXTENSIONS:
        if os.path.exists(os.path.join(IMAGE_DIR, f"{crop}.{ext}")):
            return url_for("static", filename=f"images/{crop}.{ext}")
    return None


def pretty(name):
    return name.replace("beans", " beans").replace("peas", " peas").title()


# --------------------------------------------------------------------------
# Database (SQLite): users and their prediction history
# --------------------------------------------------------------------------
HISTORY_KEYS = [f["key"] for f in FIELDS]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_execute(sql, params=()):
    conn = get_db()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def db_fetch(sql, params=()):
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def init_db():
    db_execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL)""")
    db_execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        N REAL, P REAL, K REAL,
        temperature REAL, humidity REAL, ph REAL, rainfall REAL,
        crop TEXT NOT NULL,
        confidence REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id))""")


init_db()


def now_text():
    return datetime.now().strftime("%d-%m-%Y %H:%M")


def save_prediction(user_id, values, crop, confidence):
    db_execute(
        "INSERT INTO predictions (user_id, created_at, N, P, K, temperature, "
        "humidity, ph, rainfall, crop, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, now_text(), values["N"], values["P"], values["K"],
         values["temperature"], values["humidity"], values["ph"],
         values["rainfall"], crop, confidence))


def get_rows(user_id):
    """All predictions of one user, newest first (keys match the HTML templates)."""
    return db_fetch(
        "SELECT created_at AS time, N, P, K, temperature, humidity, ph, rainfall, "
        "crop, confidence FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (user_id,))


# --------------------------------------------------------------------------
# Login helpers
# --------------------------------------------------------------------------
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    found = db_fetch("SELECT id, name, email FROM users WHERE id = ?", (uid,))
    return found[0] if found else None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            if request.path.startswith("/api/"):
                return jsonify(error="Please log in first."), 401
            return redirect(url_for("login"))
        g.user = user
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    return {"user": getattr(g, "user", None)}


# --------------------------------------------------------------------------
# Routes: login / register / logout
# --------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not name or not email or not password:
            error = "Please fill in all the fields."
        elif "@" not in email or "." not in email:
            error = "Enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif db_fetch("SELECT id FROM users WHERE email = ?", (email,)):
            error = "This email is already registered. Please log in."
        else:
            uid = db_execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, generate_password_hash(password), now_text()))
            session.clear()
            session["user_id"] = uid
            return redirect(url_for("home"))
    return render_template("auth.html", mode="register", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("home"))
    error = None
    message = "Password changed. Please log in." if request.args.get("reset") else None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        found = db_fetch("SELECT id, password_hash FROM users WHERE email = ?", (email,))
        if found and check_password_hash(found[0]["password_hash"], password):
            session.clear()
            session["user_id"] = found[0]["id"]
            return redirect(url_for("home"))
        error = "Invalid email or password."
    return render_template("auth.html", mode="login", error=error, message=message)


# --------------------------------------------------------------------------
# Forgot password (token link, valid for 30 minutes, works only once)
# --------------------------------------------------------------------------
RESET_MAX_AGE = 1800
# Optional: to send real emails set these (Gmail app password). Otherwise the
# reset link is shown on the page and printed in the terminal (demo mode).
MAIL_USER = os.environ.get("MAIL_USER", "")
MAIL_PASS = os.environ.get("MAIL_PASS", "")


def reset_serializer():
    return URLSafeTimedSerializer(app.secret_key, salt="password-reset")


def send_reset_email(to_email, name, link):
    """Send the reset link by Gmail. Returns True only if the mail was sent."""
    if not (MAIL_USER and MAIL_PASS):
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = "Reset your Smart Crop password"
        msg["From"] = MAIL_USER
        msg["To"] = to_email
        msg.set_content(
            f"Hi {name},\n\nOpen this link to set a new password (valid for 30 minutes):\n"
            f"{link}\n\nIf you did not ask for this, you can ignore this email.")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as smtp:
            smtp.login(MAIL_USER, MAIL_PASS)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        print("MAIL ERROR:", repr(exc))
        return False


@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    message = error = demo_link = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if "@" not in email:
            error = "Enter a valid email address."
        else:
            found = db_fetch("SELECT id, name, password_hash FROM users WHERE email = ?", (email,))
            if found:
                token = reset_serializer().dumps({"e": email, "h": found[0]["password_hash"][-10:]})
                link = url_for("reset_password", token=token, _external=True)
                if send_reset_email(email, found[0]["name"], link):
                    message = "Reset link sent. Check your email (valid for 30 minutes)."
                else:
                    print("PASSWORD RESET LINK:", link)
                    demo_link = link
                    message = "Email sending is not set up, so here is your reset link (valid for 30 minutes)."
            else:
                message = "If this email is registered, a reset link has been sent."
    return render_template("auth.html", mode="forgot", error=error,
                           message=message, demo_link=demo_link)


@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        data = reset_serializer().loads(token, max_age=RESET_MAX_AGE)
    except SignatureExpired:
        return render_template("auth.html", mode="forgot",
                               error="This reset link has expired. Please request a new one.")
    except BadSignature:
        return render_template("auth.html", mode="forgot", error="This reset link is not valid.")

    found = db_fetch("SELECT id, password_hash FROM users WHERE email = ?", (data.get("e", ""),))
    if not found or found[0]["password_hash"][-10:] != data.get("h"):
        return render_template("auth.html", mode="forgot",
                               error="This reset link is not valid or was already used.")

    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            db_execute("UPDATE users SET password_hash = ? WHERE id = ?",
                       (generate_password_hash(password), found[0]["id"]))
            session.clear()
            return redirect(url_for("login", reset=1))
    return render_template("auth.html", mode="reset", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# Routes: pages and prediction API
# --------------------------------------------------------------------------
@app.route("/")
@login_required
def home():
    gallery = [
        {"key": key, "name": pretty(key), "emoji": info["emoji"], "image": crop_image_url(key), "season": info["season"]}
        for key, info in CROPS.items()
    ]
    return render_template("index.html", fields=FIELDS, presets=PRESETS, gallery=gallery)


@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    payload = request.get_json(silent=True) or request.form
    values = {}

    for field in FIELDS:
        raw = payload.get(field["key"], "")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return jsonify(error=f"{field['label']} must be a number."), 400
        low, high = field["hard"]
        if not low <= number <= high:
            return jsonify(error=f"{field['label']} must be between {low} and {high}."), 400
        values[field["key"]] = number

    # DataFrame keeps the same column names the model was trained with
    frame = pd.DataFrame([values], columns=[f["key"] for f in FIELDS])

    probabilities = model.predict_proba(frame)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)[:3]

    best_crop = str(ranked[0][0])
    info = CROPS.get(best_crop, DEFAULT_INFO)

    # Save to the database under the logged-in user (never break the page if saving fails)
    try:
        save_prediction(g.user["id"], values, pretty(best_crop),
                        round(float(ranked[0][1]) * 100, 1))
    except sqlite3.Error as exc:
        print("DB ERROR:", repr(exc))

    return jsonify(
        crop=best_crop,
        name=pretty(best_crop),
        emoji=info["emoji"],
        image=crop_image_url(best_crop),
        confidence=round(float(ranked[0][1]) * 100, 1),
        fertilizer=info["fertilizer"],
        season=info["season"],
        soil=info["soil"],
        description=info["description"],
        alternatives=[
            {"name": pretty(str(name)), "emoji": CROPS.get(str(name), DEFAULT_INFO)["emoji"],
             "percent": round(float(score) * 100, 1)}
            for name, score in ranked
        ],
    )


@app.route("/history")
@login_required
def history():
    rows = get_rows(g.user["id"])[:50]
    return render_template("history.html", rows=rows)


@app.route("/performance")
@login_required
def performance():
    charts = [
        ("Model comparison", "charts/comparison.png"),
        ("Confusion matrix (Random Forest)", "charts/confusion_matrix.png"),
        ("Feature importance", "charts/feature_importance.png"),
    ]
    available = [(t, f) for t, f in charts
                 if os.path.exists(os.path.join(BASE_DIR, "static", f))]
    return render_template("performance.html", charts=available)


@app.route("/api/weather")
@login_required
def weather():
    """Live temperature and humidity for a city (Open-Meteo, no API key needed)."""
    import ssl
    from urllib.request import Request
    ctx = ssl._create_unverified_context()

    def get_json(url):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return json.load(urlopen(req, timeout=15, context=ctx))

    city = request.args.get("city", "").strip()
    if not city:
        return jsonify(error="Enter a city name."), 400
    try:
        geo = get_json("https://geocoding-api.open-meteo.com/v1/search?name="
                       + quote(city) + "&count=1")
        if not geo.get("results"):
            return jsonify(error="City not found."), 404
        place = geo["results"][0]
        cur = get_json("https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}"
                       "&current=temperature_2m,relative_humidity_2m"
                       .format(place["latitude"], place["longitude"]))["current"]
    except Exception as e:
        print("WEATHER ERROR:", repr(e))
        return jsonify(error="Could not fetch weather: " + type(e).__name__), 502
    return jsonify(city=place["name"],
                   temperature=cur["temperature_2m"],
                   humidity=cur["relative_humidity_2m"])


# Test accuracy (%) of the 5 models, from compare_models.py
MODEL_SCORES = {
    "Random Forest": 99.55,
    "Naive Bayes": 99.55,
    "SVM": 98.41,
    "Decision Tree": 97.95,
    "KNN": 97.95,
}


@app.route("/dashboard")
@login_required
def dashboard():
    rows = get_rows(g.user["id"])

    counts, confidences = {}, []
    for r in rows:
        counts[r["crop"]] = counts.get(r["crop"], 0) + 1
        confidences.append(float(r["confidence"]))
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]

    stats = {
        "total": len(rows),
        "top_crop": top[0][0] if top else "-",
        "avg_conf": round(sum(confidences) / len(confidences), 1) if confidences else 0,
        "accuracy": MODEL_SCORES["Random Forest"],
    }

    dataset = {"rows": 0, "crops": 0, "averages": []}
    csv_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        dataset["rows"] = int(len(df))
        dataset["crops"] = int(df["label"].nunique())
        for f in FIELDS:
            if f["key"] in df.columns:
                dataset["averages"].append(
                    {"label": f["label"], "unit": f["unit"],
                     "value": round(float(df[f["key"]].mean()), 1)})

    return render_template(
        "dashboard.html",
        stats=stats,
        top_labels=[t[0] for t in top],
        top_values=[t[1] for t in top],
        recent=rows[:5],
        dataset=dataset,
        model_labels=list(MODEL_SCORES.keys()),
        model_values=list(MODEL_SCORES.values()),
    )


@app.route("/report")
@login_required
def report():
    rows = get_rows(g.user["id"])

    total = len(rows)
    groups, all_conf = {}, []
    for r in rows:
        conf = float(r["confidence"])
        all_conf.append(conf)
        grp = groups.setdefault(r["crop"], {"count": 0, "conf": 0.0})
        grp["count"] += 1
        grp["conf"] += conf

    summary = [
        {"crop": crop, "count": grp["count"],
         "avg_conf": round(grp["conf"] / grp["count"], 1),
         "share": round(grp["count"] * 100 / total, 1)}
        for crop, grp in sorted(groups.items(), key=lambda kv: kv[1]["count"], reverse=True)
    ]

    avg_inputs = []
    for fld in FIELDS:
        vals = [float(r[fld["key"]]) for r in rows if r[fld["key"]] is not None]
        if vals:
            avg_inputs.append({"label": fld["label"], "unit": fld["unit"],
                               "value": round(sum(vals) / len(vals), 1)})

    return render_template(
        "report.html",
        total=total,
        summary=summary,
        avg_inputs=avg_inputs,
        top_crop=summary[0]["crop"] if summary else "-",
        avg_conf=round(sum(all_conf) / len(all_conf), 1) if all_conf else 0,
        unique_crops=len(summary),
        rows=rows[:100],
        generated=datetime.now().strftime("%d %b %Y, %H:%M"),
    )


@app.route("/report/download")
@login_required
def report_download():
    rows = get_rows(g.user["id"])
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["time"] + HISTORY_KEYS + ["crop", "confidence"])
    for r in rows:
        writer.writerow([r["time"]] + [r[k] for k in HISTORY_KEYS] + [r["crop"], r["confidence"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=crop_predictions_report.csv"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
