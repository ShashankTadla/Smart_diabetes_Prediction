
"""Compare diabetes prediction models using reliable evaluation metrics."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "dataset.csv")

NUMERIC = [
    "BMI", "Age", "Stress_Score", "Sleep_Hours"
]

CATEGORICAL = [
    "Gender", "Diet_Type", "Exercise_Frequency",
    "Heredity", "Smoking", "Alcohol"
]

FEATURES = NUMERIC + CATEGORICAL
TARGET = "Diabetes_Status"

# Keep rows with valid outcome labels
df = df.dropna(subset=[TARGET])
df = df[df[TARGET].isin([0, 1])].copy()

X = df[FEATURES]
y = df[TARGET].astype(int)

print("Total usable records:", len(df))
print("\nClass distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


def build_pipeline(classifier):
    preprocessing = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            NUMERIC
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            CATEGORICAL
        )
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier)
    ])


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, classifier in models.items():

    print("\nEvaluating:", name)

    pipeline = build_pipeline(classifier)

    # Cross-validation on training data only
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=1
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision Class 1": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall Class 1": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Recall Class 0": recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "F1 Class 1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
        "CV Balanced Accuracy Mean": cv_scores.mean(),
        "CV Balanced Accuracy Std": cv_scores.std(),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    results.append(result)

    print("Balanced accuracy:", round(result["Balanced Accuracy"], 3))
    print("PR-AUC:", round(result["PR-AUC"], 3))
    print("5-fold CV balanced accuracy:",
          round(cv_scores.mean(), 3))
    print("Confusion matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Status 0", "Status 1"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        REPORTS / f"{filename}_confusion_matrix.png",
        dpi=150
    )
    plt.close()


results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    REPORTS / "final_model_comparison.csv",
    index=False
)

# Create comparison chart
metrics = [
    "Balanced Accuracy",
    "ROC-AUC",
    "PR-AUC",
    "CV Balanced Accuracy Mean"
]

ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    rot=0
)

ax.set_title("Diabetes Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(
    REPORTS / "model_comparison_chart.png",
    dpi=150
)
plt.close()

print("\nReports saved in the 'reports' folder.")
print("Existing website model was NOT changed.")
