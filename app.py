
"""Diabetes prediction dashboard with what-if simulation."""
from pathlib import Path
import numpy as np
import shap
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "diabetes_model.joblib")
model = bundle["model"]
NUMERIC = bundle["numeric"]
CATEGORICAL = bundle["categorical"]
FEATURES = bundle["features"]

OPTIONS = {
    "Gender": ["Male", "Female"],
    "Diet_Type": ["Healthy", "Moderate", "Unhealthy"],
    "Exercise_Frequency": [
        "Daily", "3-5 times/week", "1-2 times/week", "Rarely"
    ],
    "Heredity": ["Yes", "No"],
    "Smoking": ["Yes", "No"],
    "Alcohol": ["Low", "Moderate", "High"],
}

# SHAP explainer for the trained Logistic Regression model
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

# Names of the numeric and one-hot encoded features
SHAP_FEATURES = preprocessor.get_feature_names_out()

explainer = shap.LinearExplainer(
    classifier,
    np.zeros((1, len(SHAP_FEATURES)))
)



def explain_prediction(values):
    patient = pd.DataFrame([values], columns=FEATURES)

    transformed = preprocessor.transform(patient)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer(transformed).values[0]

    contributions = []

    for name, value in zip(SHAP_FEATURES, shap_values):
        readable_name = (
            name.replace("numeric__", "")
                .replace("categorical__", "")
                .replace("_", " ")
        )

        contributions.append({
            "name": readable_name,
            "value": round(float(value), 3),
            "direction": "Higher" if value > 0 else "Lower",
            "strength": abs(float(value))
        })

    contributions.sort(
        key=lambda item: item["strength"],
        reverse=True
    )

    top = contributions[:5]

    maximum = max(
        (item["strength"] for item in top),
        default=0
    )

    for item in top:
        if maximum > 0:
            item["width"] = round(
                item["strength"] / maximum * 100, 1
            )
        else:
            item["width"] = 0

    return top


def make_recommendations(v):
    tips = []

    if v["BMI"] >= 25:
        tips.append("Choose balanced meals and regular movement.")
    if v["Exercise_Frequency"] in ("Rarely", "1-2 times/week"):
        tips.append("Gradually increase physical activity.")
    if v["Diet_Type"] == "Unhealthy":
        tips.append("Add more vegetables and whole grains to meals.")
    if v["Smoking"] == "Yes":
        tips.append("Consider support to stop smoking.")
    if v["Sleep_Hours"] < 7:
        tips.append("Try to maintain a consistent sleep schedule.")
    if v["Stress_Score"] >= 7:
        tips.append("Practice stress-management techniques.")
    if v["Heredity"] == "Yes":
        tips.append("Discuss family history and screening with a clinician.")

    if not tips:
        tips.append("Continue healthy habits and appropriate screening.")

    return tips


def validate(values):
    if not (
        10 <= values["BMI"] <= 80
        and 18 <= values["Age"] <= 120
        and 0 <= values["Stress_Score"] <= 10
        and 0 <= values["Sleep_Hours"] <= 24
    ):
        raise ValueError("Check the numeric input ranges.")

    if any(values[c] not in OPTIONS[c] for c in CATEGORICAL):
        raise ValueError("Invalid dropdown selection.")


def predict(values):
    patient = pd.DataFrame([values], columns=FEATURES)
    return round(float(model.predict_proba(patient)[0][1]) * 100, 1)


def personalized_lifestyle_guide(values, explanations):
    guide = []

    def add(feature, title, explanation, actions, priority=0):
        guide.append({
            "feature": feature,
            "title": title,
            "explanation": explanation,
            "actions": actions,
            "priority": priority
        })

    # Identify the strongest SHAP contributors.
    # SHAP values represent model influence, not causal effects.
    feature_strength = {}

    for item in explanations:
        name = item["name"].lower()
        strength = abs(float(item["value"]))

        for feature in FEATURES:
            readable = feature.replace("_", " ").lower()

            if name == readable or name.startswith(readable + " "):
                feature_strength[feature] = max(
                    feature_strength.get(feature, 0),
                    strength
                )

    def priority(feature):
        return feature_strength.get(feature, 0)

    bmi = float(values["BMI"])
    sleep = float(values["Sleep_Hours"])
    stress = float(values["Stress_Score"])

    if bmi >= 25:
        add(
            "BMI",
            "Build healthy weight-management habits",
            f"Your BMI is {bmi:.1f}. BMI is a screening measure, "
            "not a complete assessment of your health.",
            [
                "Choose balanced portions and include vegetables, "
                "whole grains, and protein-rich foods.",
                "Start with comfortable daily movement and increase gradually.",
                "Avoid crash diets; focus on habits you can maintain."
            ],
            priority("BMI")
        )

    if sleep < 7:
        add(
            "Sleep_Hours",
            "Improve your sleep routine",
            f"You reported {sleep:g} hours of sleep per night.",
            [
                "Aim for 7–9 hours of sleep if you are an adult.",
                "Keep a consistent sleep and wake-up schedule.",
                "Reduce caffeine late in the day and limit screens before bed."
            ],
            priority("Sleep_Hours")
        )

    if stress >= 7:
        add(
            "Stress_Score",
            "Create a daily stress-management routine",
            f"Your reported stress score is {stress:g}/10.",
            [
                "Try 5–10 minutes of breathing exercises or mindfulness.",
                "Include short walks and regular breaks in your day.",
                "Speak with someone you trust or a professional if stress persists."
            ],
            priority("Stress_Score")
        )

    exercise = values["Exercise_Frequency"]

    if exercise in ["Rarely", "1-2 times/week"]:
        add(
            "Exercise_Frequency",
            "Increase physical activity gradually",
            f"Your current exercise frequency is {exercise}.",
            [
                "Begin with a 10–15 minute walk on most days.",
                "Gradually work toward 150 minutes of moderate activity weekly.",
                "Include muscle-strengthening activities twice a week "
                "when appropriate for your health."
            ],
            priority("Exercise_Frequency")
        )

    diet = values["Diet_Type"]

    if diet in ["Unhealthy", "Moderate"]:
        add(
            "Diet_Type",
            "Make your meals more balanced",
            f"You described your diet as {diet.lower()}.",
            [
                "Add vegetables, legumes, and fiber-rich foods regularly.",
                "Replace sugary drinks with water or unsweetened alternatives.",
                "Choose minimally processed foods more often."
            ],
            priority("Diet_Type")
        )

    if values["Smoking"] == "Yes":
        add(
            "Smoking",
            "Work toward stopping smoking",
            "Smoking is a modifiable health risk factor.",
            [
                "Set a realistic quit plan.",
                "Seek support from a healthcare professional or quit service.",
                "Avoid secondhand smoke when possible."
            ],
            priority("Smoking")
        )

    if values["Alcohol"] in ["Moderate", "High"]:
        add(
            "Alcohol",
            "Review alcohol consumption",
            f"You selected {values['Alcohol'].lower()} alcohol consumption.",
            [
                "Consider reducing alcohol consumption.",
                "Include alcohol-free days.",
                "Discuss safer limits with a healthcare professional."
            ],
            priority("Alcohol")
        )

    if values["Heredity"] == "Yes":
        add(
            "Heredity",
            "Be proactive about family history",
            "You reported a family history of diabetes. "
            "This cannot be changed, but awareness can support prevention.",
            [
                "Discuss your family history with a healthcare professional.",
                "Ask whether diabetes screening is appropriate for you.",
                "Continue building sustainable nutrition and activity habits."
            ],
            priority("Heredity")
        )

    # Put the strongest model contributors first.
    # Other relevant lifestyle factors remain visible.
    guide.sort(
        key=lambda item: item["priority"],
        reverse=True
    )

    for index, item in enumerate(guide):
        item["rank"] = index + 1
        item["top_contributor"] = item["priority"] > 0

    if not guide:
        add(
            "General",
            "Maintain your healthy routines",
            "Your reported lifestyle factors do not trigger "
            "the specific improvement suggestions in this guide.",
            [
                "Maintain regular physical activity.",
                "Continue balanced eating and consistent sleep.",
                "Follow recommended preventive health screening."
            ]
        )

    return guide

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    simulated_probability = None
    tips = []
    values = {}
    explanations = []
    lifestyle_guide = []
    error = None

    if request.method == "POST":
        try:
            values = {c: float(request.form[c]) for c in NUMERIC}
            values.update({c: request.form[c] for c in CATEGORICAL})
            validate(values)

            probability = predict(values)
            result = (
                "Higher model-estimated likelihood"
                if probability >= 50
                else "Lower model-estimated likelihood"
            )
            tips = make_recommendations(values)
            explanations = explain_prediction(values)
            lifestyle_guide = personalized_lifestyle_guide(
    values,
    explanations
)

            # What-if scenario: changes only selected lifestyle inputs.
            simulated = values.copy()
            simulated["BMI"] = float(
                request.form.get("sim_BMI") or values["BMI"]
            )
            simulated["Sleep_Hours"] = float(
                request.form.get("sim_Sleep_Hours")
                or values["Sleep_Hours"]
            )
            simulated["Stress_Score"] = float(
                request.form.get("sim_Stress_Score")
                or values["Stress_Score"]
            )
            simulated["Diet_Type"] = (
                request.form.get("sim_Diet_Type") or values["Diet_Type"]
            )
            simulated["Exercise_Frequency"] = (
                request.form.get("sim_Exercise_Frequency")
                or values["Exercise_Frequency"]
            )

            validate(simulated)
            simulated_probability = predict(simulated)

        except (ValueError, KeyError) as exc:
            error = f"Please check your entries: {exc}"

    return render_template(
        "index.html",
        numeric=NUMERIC,
        options=OPTIONS,
        values=values,
        result=result,
        probability=probability,
        simulated_probability=simulated_probability,
        tips=tips,
        lifestyle_guide=lifestyle_guide,
        error=error,
        explanations=explanations,
    )


if __name__ == "__main__":
    app.run(debug=True)
