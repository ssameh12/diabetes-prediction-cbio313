"""Streamlit front-end for the diabetes screening model.

Run from the repository root:

    streamlit run src/app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_frame():
    """Used as the SHAP background and for the population comparison."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from data_prep import FEATURES, clean, load_raw

    return clean(load_raw())[FEATURES]


if not MODEL_PATH.exists():
    st.error("models/best_model.joblib not found. Run `python src/train.py` first.")
    st.stop()

bundle = load_bundle()
pipeline, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]
background = load_training_frame()

st.title("🩺 Diabetes Risk Screening")
st.caption(
    f"Model: **{bundle['model_name']}** · decision threshold **{threshold:.2f}** "
    "(chosen on out-of-fold training predictions to reach 80% recall)"
)

st.warning(
    "Educational course project (CBIO313). Trained on 768 Pima Indian women aged 21+. "
    "Not a medical device and not valid for other populations.",
    icon="⚠️",
)

# --- inputs -----------------------------------------------------------------
st.sidebar.header("Patient measurements")
DEFAULTS = {
    "Pregnancies": (0, 17, 3, 1, "Number of times pregnant"),
    "Glucose": (40, 200, 120, 1, "Plasma glucose 2 h into an OGTT (mg/dL)"),
    "BloodPressure": (30, 130, 70, 1, "Diastolic blood pressure (mm Hg)"),
    "SkinThickness": (5, 100, 20, 1, "Triceps skin-fold thickness (mm)"),
    "Insulin": (10, 850, 80, 5, "2-hour serum insulin (mu U/mL)"),
    "BMI": (15.0, 70.0, 32.0, 0.1, "Body mass index (kg/m²)"),
    "DiabetesPedigreeFunction": (0.05, 2.5, 0.45, 0.01, "Family-history diabetes score"),
    "Age": (21, 90, 35, 1, "Age (years)"),
}

values = {}
for name, (lo, hi, default, step, help_text) in DEFAULTS.items():
    values[name] = st.sidebar.slider(name, lo, hi, default, step, help=help_text)

patient = pd.DataFrame([values])[features]

# --- prediction -------------------------------------------------------------
probability = float(pipeline.predict_proba(patient)[0, 1])
flagged = probability >= threshold

left, right = st.columns([1, 1.4])

with left:
    st.subheader("Result")
    st.metric("Predicted probability of diabetes", f"{probability:.1%}")
    if flagged:
        st.error(f"**Flagged for confirmatory testing** — probability is at or above {threshold:.0%}.")
    else:
        st.success(f"**Not flagged** — probability is below the {threshold:.0%} screening threshold.")
    st.progress(min(probability, 1.0))
    st.caption(
        "The threshold sits well below 50% on purpose: in screening, missing a diabetic costs more "
        "than sending a healthy patient for one extra blood test."
    )

with right:
    st.subheader("How this patient compares")
    comparison = pd.DataFrame(
        {
            "patient": [values[f] for f in features],
            "population median": background[features].median().round(2).values,
            "percentile": [
                round(float((background[f].dropna() < values[f]).mean() * 100), 1) for f in features
            ],
        },
        index=features,
    )
    st.dataframe(comparison, width="stretch")

# --- explanation ------------------------------------------------------------
st.subheader("Why the model said that")

try:
    import shap

    transformed = pd.DataFrame(
        pipeline.named_steps["scale"].transform(
            pipeline.named_steps["impute"].transform(patient)),
        columns=features,
    )
    bg = pd.DataFrame(
        pipeline.named_steps["scale"].transform(
            pipeline.named_steps["impute"].transform(background.sample(200, random_state=42))),
        columns=features,
    )
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        shap_values = shap.TreeExplainer(model).shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif np.ndim(shap_values) == 3:
            shap_values = shap_values[:, :, 1]
    elif hasattr(model, "coef_"):
        shap_values = shap.LinearExplainer(model, bg).shap_values(transformed)
    else:
        explainer = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], shap.kmeans(bg, 25))
        shap_values = explainer.shap_values(transformed, nsamples=200, silent=True)

    contributions = pd.Series(np.asarray(shap_values).ravel(), index=features).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#c44e52" if v > 0 else "#4c72b0" for v in contributions.values]
    ax.barh(contributions.index, contributions.values, color=colours)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("← lowers risk        contribution to this prediction        raises risk →")
    ax.set_title("SHAP contribution per measurement, for this patient")
    st.pyplot(fig)
    plt.close(fig)

    top = contributions.abs().sort_values(ascending=False).head(3).index.tolist()
    st.caption(f"Largest influences on this prediction: {', '.join(top)}.")
except Exception as exc:  # noqa: BLE001 - the app must still work without shap
    st.info(f"Per-patient explanation unavailable ({exc}).")
