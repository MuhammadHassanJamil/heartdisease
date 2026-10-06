"""
Retrain and evaluate the Heart Disease SVC model.

- Trains ONLY on training-data.csv (hyperparameters chosen by 5-fold CV on the
  training data, so the test set is never used for tuning).
- Evaluates ONCE on the held-out testing-data.csv.
- Saves the fitted pipeline (StandardScaler + SVC) to svc_trained_model.pkl in
  the project root and in streamlit-app/.
"""

import pickle
import shutil
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent
FEATURES = ["cp", "ca", "thalach", "oldpeak"]
TARGET = "target"

train = pd.read_csv(ROOT / "training-data.csv")
test = pd.read_csv(ROOT / "testing-data.csv")
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print("Train class balance:", y_train.value_counts().to_dict())

# Baseline: the old approach (plain SVC, no scaling)
baseline = SVC().fit(X_train, y_train)
print(f"\nBaseline SVC (no scaling) test accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")

# Properly trained model: scaling + tuned SVC, tuned with CV on training data only
pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
param_grid = {
    "svc__kernel": ["rbf", "linear"],
    "svc__C": [0.1, 0.5, 1, 5, 10, 50],
    "svc__gamma": ["scale", 0.01, 0.05, 0.1, 0.5],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
model = search.best_estimator_

print("\nBest params:", search.best_params_)
print(f"5-fold CV accuracy on training data: {search.best_score_:.3f}")
print(f"Training accuracy: {accuracy_score(y_train, model.predict(X_train)):.3f}")

pred = model.predict(X_test)
print(f"\nTEST accuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=["No Heart Disease (0)", "Heart Disease (1)"]))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, pred))

# Sanity check on the extreme profile that the deployed app got wrong
probe = pd.DataFrame({"cp": [3], "ca": [4], "thalach": [205], "oldpeak": [6.4]})
print("\nSanity probes (cp, ca, thalach, oldpeak) -> prediction:")
print("  3, 4, 205, 6.4 ->", model.predict(probe)[0])
healthy = pd.DataFrame({"cp": [2], "ca": [0], "thalach": [175], "oldpeak": [0.0]})
print("  2, 0, 175, 0.0 ->", model.predict(healthy)[0])
risky = pd.DataFrame({"cp": [0], "ca": [2], "thalach": [120], "oldpeak": [2.8]})
print("  0, 2, 120, 2.8 ->", model.predict(risky)[0])

# Save model
out = ROOT / "svc_trained_model.pkl"
with open(out, "wb") as f:
    pickle.dump(model, f)
shutil.copy(out, ROOT / "streamlit-app" / "svc_trained_model.pkl")
pd.DataFrame({**X_test.to_dict("list"), "target": y_test, "Predictions": pred}).to_csv(
    ROOT / "model-predictions.csv", index=False
)
print("\nSaved model to svc_trained_model.pkl and streamlit-app/svc_trained_model.pkl")
