from flask import Flask, render_template, request
import pandas as pd
import joblib
import random
import re
import traceback

app = Flask(__name__)

# --- Config / paths ---
TRAIN_CSV = "training.csv"            # your original training CSV (must exist)
MODEL_PKL = "disease_model.pkl"       # your saved model
LABEL_ENCODER_PKL = "label_encoder.pkl"

# --- Load model & encoder ---
model = joblib.load(MODEL_PKL)
le = joblib.load(LABEL_ENCODER_PKL)

# --- Load training CSV columns to get exact feature names used at training time ---
# (we assume the CSV used to train the model is available locally)
df_train = pd.read_csv(TRAIN_CSV)
if "prognosis" in df_train.columns:
    feature_cols = [c for c in df_train.columns if c != "prognosis"]
else:
    # If your training CSV uses a different target name, adjust here or keep full columns
    feature_cols = list(df_train.columns)

# For safety: if the model exposes feature_names_in_ (sklearn >= 1.0),
# prefer that exact list (it is the truth the model expects).
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    # If model_features and feature_cols differ, prefer model_features (model is authoritative)
    if set(model_features) != set(feature_cols):
        print("Warning: model.feature_names_in_ differs from CSV columns. Using model.feature_names_in_.")
        feature_cols = model_features

all_symptoms = feature_cols.copy()  # exact feature names used by model

# --- Helper: user-friendly labels for display (keeps backend values exact) ---
def friendly_label(feature_name: str) -> str:
    # convert "spotting_ urination" -> "Spotting urination", remove weird parentheses, extra spaces
    s = feature_name
    s = s.replace("_", " ")
    # remove multiple spaces
    s = re.sub(r"\s+", " ", s)
    # remove outer parentheses characters (but keep inner content)
    s = s.replace("(", "").replace(")", "")
    s = s.strip()
    # capitalize first letter of each word for nicer display
    s = " ".join(word.capitalize() for word in s.split(" "))
    return s

# --- Categorize symptoms heuristically (works with exact feature names) ---
def categorize(features):
    skin_kw = ["skin", "rash", "peel", "blister", "blackhead", "spot", "crust", "scurr"]
    resp_kw = ["cough", "sneez", "phlegm", "throat", "breath", "nose", "sinus", "chest"]
    dig_kw = ["stomach", "abdominal", "nausea", "vomit", "indigest", "diarr", "constip", "belly", "ulcer", "acidity"]
    mus_kw = ["joint", "muscle", "back", "knee", "hip", "neck", "cramp", "stiff", "movement"]
    # build lists
    skin, resp, dig, mus = [], [], [], []
    for f in features:
        lower = f.lower()
        if any(k in lower for k in skin_kw):
            skin.append(f)
        elif any(k in lower for k in resp_kw):
            resp.append(f)
        elif any(k in lower for k in dig_kw):
            dig.append(f)
        elif any(k in lower for k in mus_kw):
            mus.append(f)
    # general = remaining
    others = [f for f in features if f not in (skin + resp + dig + mus)]
    categorized = {
        "Skin": skin,
        "Respiratory": resp,
        "Digestive": dig,
        "Muscular/Joint": mus,
        "General": others
    }
    return categorized

categorized_symptoms = categorize(all_symptoms)

# --- Health tips (random) ---
tips = [
    "Drink plenty of water daily.",
    "Exercise at least 30 minutes a day.",
    "Eat more fruits and vegetables.",
    "Get 7-8 hours of sleep every night.",
    "Take breaks to reduce stress.",
    "Wash your hands regularly to prevent infections.",
    "Maintain good posture to avoid back and neck pain."
]

# --- Flask route ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    tip = None
    error_message = None
    try:
        if request.method == "POST":
            # collect selected symptom for each category (exact backend values)
            selected = []
            for cat in categorized_symptoms.keys():
                # the form fields are named by category (e.g., "Skin", "Respiratory", ...)
                val = request.form.get(cat)
                if val:
                    selected.append(val)

            # build input DataFrame using exact feature names the model expects
            X_input = pd.DataFrame([{f: (1 if f in selected else 0) for f in all_symptoms}])
            # predict
            pred_encoded = model.predict(X_input)[0]
            prediction = le.inverse_transform([pred_encoded])[0]
            tip = random.choice(tips)
    except Exception as e:
        # capture exception and show friendly message in UI (but not sensitive internals)
        error_message = "Prediction failed: model input mismatch. See console for details."
        print("----- Prediction error -----")
        traceback.print_exc()

    # prepare a display-friendly categorized_symptoms: a mapping category -> list of (value, label)
    categorized_display = {}
    for cat, feats in categorized_symptoms.items():
        categorized_display[cat] = [(f, friendly_label(f)) for f in feats]

    return render_template(
        "index.html",
        categorized_symptoms=categorized_display,
        prediction=prediction,
        tip=tip,
        error_message=error_message,
        app_name="TeleCure"
    )

if __name__ == "__main__":
    app.run(debug=True)
