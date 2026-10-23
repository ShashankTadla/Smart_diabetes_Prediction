# Smart Diabetes Prediction (no Glucose / Insulin)

This replaces the previous Pima model and web form with the uploaded 5,000-row dataset.

1. Copy all files into your INNER diabetes_project_starter folder, replacing app.py, train_model.py, templates/index.html, static/style.css, requirements.txt, and dataset.csv.
2. Stop Flask with Ctrl+C in VS Code terminal.
3. With (.venv) active, run `python -m pip install -r requirements.txt` if necessary.
4. Run `python train_model.py` to regenerate `diabetes_model.joblib`.
5. Run `python app.py` and open http://127.0.0.1:5000.

Glucose and Insulin are excluded from the model features and UI. Missing inputs in training are imputed within the pipeline after the train/test split. The model's class balance is skewed toward status 1. This is an educational prototype, not a diagnosis or a clinically calibrated risk estimate.
