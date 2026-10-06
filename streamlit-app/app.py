"""
Heart Disease Prediction - Student Demo App
--------------------------------------------
A small Streamlit web app that lets students interact with the trained
Support Vector Classifier from the "heartdisease.ipynb" notebook, without
needing to install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook. All four
input features (cp, ca, thalach, oldpeak) are already numeric, so no
encoding is needed: the values from the form go straight to the model in
the same column order used during training.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

# Column order must match the training data exactly
FEATURE_COLUMNS = ["cp", "ca", "thalach", "oldpeak"]

CHEST_PAIN_TYPES = {
    0: "0 - Typical angina",
    1: "1 - Atypical angina",
    2: "2 - Non-anginal pain",
    3: "3 - Asymptomatic",
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("🫀 Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction**
    notebook. Enter a patient's clinical values below and the trained Support
    Vector Classifier (SVC) will predict whether heart disease is present.

    *Model accuracy on held-out test data: **81%** (303-row dataset, 202 train and 101 test).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=list(CHEST_PAIN_TYPES.keys()),
        format_func=lambda value: CHEST_PAIN_TYPES[value],
        index=2,
    )
    thalach = st.slider("Max Heart Rate Achieved (thalach)", min_value=70, max_value=205, value=150)
with col2:
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3, 4], index=0)
    oldpeak = st.slider(
        "ST Depression Induced by Exercise (oldpeak)",
        min_value=0.0,
        max_value=6.5,
        value=1.0,
        step=0.1,
    )

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    features = pd.DataFrame(
        {"cp": [cp], "ca": [ca], "thalach": [thalach], "oldpeak": [oldpeak]}
    )[FEATURE_COLUMNS]

    prediction = model.predict(features)[0]

    st.divider()
    if prediction == 1:
        st.error("### ❌ HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(features, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from a small heart disease dataset. "
    "Predictions reflect patterns in this dataset only and are NOT medical advice."
)
