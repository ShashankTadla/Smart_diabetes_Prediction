"""Train a diabetes-status classifier without glucose or insulin."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = ['BMI', 'Age', 'Stress_Score', 'Sleep_Hours']
CATEGORICAL = ['Gender', 'Diet_Type', 'Exercise_Frequency', 'Heredity', 'Smoking', 'Alcohol']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'Diabetes_Status'


def main():
    df = pd.read_csv(ROOT / 'dataset.csv')
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing columns: {missing}')
    df = df.dropna(subset=[TARGET]).copy()
    if not df[TARGET].isin([0, 1]).all():
        raise SystemExit('Diabetes_Status must contain 0 or 1.')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ('preprocessing', prep),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print('Training rows:', len(X_train), '| Test rows:', len(X_test))
    print('Test-set class distribution:', y_test.value_counts().to_dict())
    print(classification_report(y_test, preds, zero_division=0))
    print('Confusion matrix [[TN, FP], [FN, TP]]:')
    print(confusion_matrix(y_test, preds))
    print(f'ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}')
    joblib.dump({'model': model, 'features': FEATURES,
                 'numeric': NUMERIC, 'categorical': CATEGORICAL},
                ROOT / 'diabetes_model.joblib')
    print('Saved diabetes_model.joblib')


if __name__ == '__main__':
    main()
