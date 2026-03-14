"""
app.py
------
Streamlit web application for the Heart Disease Prediction & Classification project.

Pages:
  🏠 Home          — Overview, dataset info, feature descriptions
  🔮 Predict       — Patient data input → live model inference
  📊 Model Results — All saved evaluation & comparison plots
  🚀 Run Pipeline  — Trigger the full ML pipeline with live logs
"""

import os
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import subprocess, sys, os

if os.environ.get("STREAMLIT_RUNNING") != "1":
    os.environ["STREAMLIT_RUNNING"] = "1"
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()
# ── Path setup so we can import src/ modules ─────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
sys.path.insert(0, SRC_DIR)

from preprocessing import NUMERICAL_COLS, CATEGORICAL_COLS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="❤️ Heart Disease AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #c62828 0%, #6a1010 50%, #1a0505 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(198,40,40,0.35);
}
.main-header h1 { color: #fff; font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.main-header p  { color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-top: 0.4rem; }

.stat-card {
    background: linear-gradient(135deg, #1e1e2e, #2a1a1a);
    border: 1px solid rgba(198,40,40,0.3);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(198,40,40,0.22); }
.stat-card .stat-num  { font-size: 2rem; font-weight: 800; color: #ef5350; }
.stat-card .stat-label{ font-size: 0.85rem; color: #aaa; margin-top: 0.2rem; }

.predict-box {
    background: linear-gradient(135deg, #1e1e2e, #16213e);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.result-healthy {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 14px; padding: 1.8rem 2rem; text-align: center;
    box-shadow: 0 4px 24px rgba(27,94,32,0.5);
}
.result-disease {
    background: linear-gradient(135deg, #b71c1c, #c62828);
    border-radius: 14px; padding: 1.8rem 2rem; text-align: center;
    box-shadow: 0 4px 24px rgba(183,28,28,0.5);
}
.result-healthy h2, .result-disease h2 { color: #fff; font-size: 1.8rem; margin: 0; }
.result-healthy p, .result-disease p   { color: rgba(255,255,255,0.85); margin-top: 0.3rem; }

.section-title {
    font-size: 1.35rem; font-weight: 700; color: #ef5350;
    border-left: 4px solid #ef5350;
    padding-left: 0.7rem; margin: 1.5rem 0 1rem 0;
}

.sidebar-nav-btn { width: 100%; }

.feature-tag {
    display: inline-block;
    background: rgba(239,83,80,0.15);
    border: 1px solid rgba(239,83,80,0.35);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #ef9a9a;
    margin: 0.15rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0505 0%, #0d0d1a 100%);
    border-right: 1px solid rgba(198,40,40,0.25);
}
[data-testid="stSidebar"] .stRadio label { color: #ddd !important; font-size: 1rem; }

div[data-testid="metric-container"] {
    background: rgba(30,30,46,0.7);
    border: 1px solid rgba(239,83,80,0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:3rem'>❤️</div>
        <div style='color:#ef5350; font-weight:800; font-size:1.1rem;'>Heart Disease AI</div>
        <div style='color:#888; font-size:0.78rem; margin-top:0.2rem;'>Cleveland Dataset · UCI ML Repo</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔮  Predict", "📊  Model Results", "🚀  Run Pipeline"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<div style='color:#666; font-size:0.75rem; text-align:center;'>5 Models · 13 Features<br>Multi-class Classification</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>AI-powered multi-class classification of Cleveland Heart Disease types</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("303", "Patient Records"),
        ("13", "Clinical Features"),
        ("5", "Disease Classes"),
        ("5", "ML Models Trained"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # About section
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        This application uses machine learning to **predict and classify heart disease types**
        from clinical patient data sourced from the **UCI ML Repository** (Cleveland dataset, ID 45).

        The pipeline trains **5 classifiers** on 13 clinical features and predicts one of 5 outcomes:

        | Class | Meaning |
        |-------|---------|
        | **0** | Healthy — No heart disease |
        | **1** | Type 1 HD — Mild stenosis |
        | **2** | Type 2 HD — Moderate disease |
        | **3** | Type 3 HD — Severe disease |
        | **4** | Type 4 HD — Critical condition |
        """)

    with col_right:
        st.markdown('<div class="section-title">Models Used</div>', unsafe_allow_html=True)
        model_info = {
            "🔵 Logistic Regression": "Fast, interpretable baseline",
            "🌲 Random Forest": "Ensemble of 200 decision trees",
            "⚡ SVM (RBF Kernel)": "High-margin classifier",
            "🔍 KNN": "Distance-weighted neighbors",
            "🚀 XGBoost": "Gradient boosted trees",
        }
        for mname, mdesc in model_info.items():
            st.markdown(f"**{mname}** — {mdesc}")

    # Feature descriptions
    st.markdown('<div class="section-title">Clinical Feature Reference</div>', unsafe_allow_html=True)
    features_data = {
        "Feature": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "Description": [
            "Age in years",
            "Sex (1 = Male, 0 = Female)",
            "Chest pain type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic)",
            "Resting blood pressure (mm Hg)",
            "Serum cholesterol (mg/dl)",
            "Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)",
            "Resting ECG results (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)",
            "Maximum heart rate achieved",
            "Exercise-induced angina (1 = Yes, 0 = No)",
            "ST depression induced by exercise relative to rest",
            "Slope of peak exercise ST segment (0 = up, 1 = flat, 2 = down)",
            "Number of major vessels colored by fluoroscopy (0–3)",
            "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)",
        ],
        "Type": ["Numerical", "Categorical", "Categorical", "Numerical", "Numerical",
                 "Categorical", "Categorical", "Numerical", "Categorical", "Numerical",
                 "Categorical", "Categorical", "Categorical"],
    }
    df_feat = pd.DataFrame(features_data)
    st.dataframe(df_feat, use_container_width=True, hide_index=True)

    # Pipeline flow
    st.markdown('<div class="section-title">Pipeline Overview</div>', unsafe_allow_html=True)
    steps = ["📥 Load Data", "🔧 Preprocess", "🚀 Train 5 Models", "📊 Evaluate", "📈 Compare"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(f"""
            <div style='background:rgba(239,83,80,0.1); border:1px solid rgba(239,83,80,0.3);
                border-radius:10px; padding:0.8rem; text-align:center; font-size:0.9rem;'>
                {step}
            </div>
            """, unsafe_allow_html=True)
        if i < len(steps) - 1:
            pass  # arrow handled by layout


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Patient Prediction</h1>
        <p>Enter clinical values and get an instant AI-powered diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    # Load available models
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "SVM":                 "svm.pkl",
        "KNN":                 "knn.pkl",
    }

    def load_model(name):
        path = os.path.join(MODELS_DIR, model_files[name])
        if os.path.exists(path):
            return joblib.load(path)
        return None

    # ── Columns layout ────────────────────────────────────────────────────────
    col_inputs, col_result = st.columns([1.1, 1])

    with col_inputs:
        st.markdown('<div class="section-title">Patient Clinical Data</div>', unsafe_allow_html=True)

        with st.form("predict_form"):
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                age      = st.slider("Age (years)", 20, 80, 55)
                trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 130)
                chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
                thalach  = st.slider("Max Heart Rate", 60, 220, 150)
                oldpeak  = st.slider("ST Depression", 0.0, 6.5, 1.0, step=0.1)

            with r1c2:
                sex     = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
                cp      = st.selectbox("Chest Pain Type", [(0,"Typical Angina"),(1,"Atypical"),(2,"Non-Anginal"),(3,"Asymptomatic")], format_func=lambda x: f"{x[0]} – {x[1]}")[0]
                fbs     = st.selectbox("Fasting Blood Sugar > 120", [(1,"Yes"),(0,"No")], format_func=lambda x: x[1])[0]
                restecg = st.selectbox("Resting ECG", [(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")], format_func=lambda x: f"{x[0]} – {x[1]}")[0]
                exang   = st.selectbox("Exercise-Induced Angina", [(1,"Yes"),(0,"No")], format_func=lambda x: x[1])[0]
                slope   = st.selectbox("ST Slope", [(0,"Up"),(1,"Flat"),(2,"Down")], format_func=lambda x: f"{x[0]} – {x[1]}")[0]
                ca      = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
                thal    = st.selectbox("Thalassemia", [(1,"Normal"),(2,"Fixed Defect"),(3,"Reversible Defect")], format_func=lambda x: f"{x[0]} – {x[1]}")[0]

            chosen_model_name = st.selectbox("🤖 Select Model", list(model_files.keys()), index=1)

            submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-title">Diagnosis Result</div>', unsafe_allow_html=True)

        if submitted:
            model = load_model(chosen_model_name)
            if model is None:
                st.error(f"Model file not found. Please run the pipeline first.")
            else:
                # Build input in the exact feature order preprocessing uses
                # NUMERICAL_COLS + CATEGORICAL_COLS
                feature_values = [age, trestbps, chol, thalach, oldpeak,  # numerical (5)
                                   cp, restecg, slope, ca, thal, sex, fbs, exang]  # categorical (8)

                # Apply mean/std scaling for numerical features using stored scaler stats
                # We replicate simple z-score with dataset-typical means/stds
                # (Approximate; exact scaler not saved — use reasonable dataset stats)
                MEAN = [54.4, 131.6, 246.7, 149.6, 1.04]
                STD  = [9.0,  17.6,  51.8,  22.9,  1.16]

                inp = np.array(feature_values, dtype=float)
                for i in range(5):
                    inp[i] = (inp[i] - MEAN[i]) / (STD[i] + 1e-8)

                inp_2d = inp.reshape(1, -1)
                pred_class = model.predict(inp_2d)[0]

                CLASS_LABELS = {0: "Healthy", 1: "Type 1 HD", 2: "Type 2 HD", 3: "Type 3 HD", 4: "Type 4 HD"}
                label = CLASS_LABELS.get(int(pred_class), str(pred_class))

                if pred_class == 0:
                    st.markdown(f"""
                    <div class="result-healthy">
                        <h2>✅ {label}</h2>
                        <p>No signs of heart disease detected based on the given clinical values.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-disease">
                        <h2>⚠️ {label}</h2>
                        <p>Heart disease detected. Please consult a cardiologist for professional evaluation.</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                # Probability chart
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(inp_2d)[0]
                    classes_present = list(model.classes_)
                    # Ensure 5 classes
                    all_classes = [0, 1, 2, 3, 4]
                    all_proba = []
                    for c in all_classes:
                        if c in classes_present:
                            all_proba.append(proba[classes_present.index(c)])
                        else:
                            all_proba.append(0.0)

                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    fig.patch.set_facecolor("#1e1e2e")
                    ax.set_facecolor("#1e1e2e")
                    colors = ["#4CAF50" if i == 0 else "#ef5350" for i in all_classes]
                    bars = ax.barh([CLASS_LABELS[c] for c in all_classes], all_proba,
                                   color=colors, height=0.55, edgecolor="none")
                    for bar, p in zip(bars, all_proba):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                f"{p*100:.1f}%", va="center", color="white", fontsize=9, fontweight="bold")
                    ax.set_xlim(0, 1.18)
                    ax.set_xlabel("Probability", color="#aaa", fontsize=9)
                    ax.tick_params(colors="white")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.tick_params(left=False)
                    ax.set_title("Prediction Probabilities", color="white", fontsize=11, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(inp_2d)[0]
                    classes_present = list(model.classes_)
                    st.markdown("**Decision Scores:**")
                    for cls, sc in zip(classes_present, scores if scores.ndim > 0 else [scores]):
                        st.write(f"  {CLASS_LABELS.get(cls, cls)}: `{sc:.3f}`")

                # Input summary
                st.markdown("**Input Summary**")
                summary_df = pd.DataFrame({
                    "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
                                "Fasting BS", "Rest ECG", "Max HR", "Exang", "ST Depression",
                                "ST Slope", "Vessels (ca)", "Thal"],
                    "Value": [age, "Male" if sex==1 else "Female", cp, trestbps, chol,
                              "Yes" if fbs==1 else "No", restecg, thalach,
                              "Yes" if exang==1 else "No", oldpeak, slope, ca, thal]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Fill in the patient data and click **Predict** to see results.")
            st.markdown("""
            <div style='color:#888; font-size:0.85rem; margin-top:1rem;'>
            <b>Disclaimer:</b> This tool is for educational purposes only.
            Always consult a qualified medical professional for clinical decisions.
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Results":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Results</h1>
        <p>Evaluation metrics, confusion matrices, ROC curves and model comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    models_list = ["Logistic Regression", "SVM", "KNN"]
    slug = lambda n: n.replace(" ", "_").lower()

    tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrices", "📈 ROC Curves", "⚖️ Model Comparison"])

    with tab1:
        st.markdown('<div class="section-title">Confusion Matrices — All Models</div>', unsafe_allow_html=True)
        rows = [models_list]
        for row_models in rows:
            cols = st.columns(len(row_models))
            for col, mname in zip(cols, row_models):
                with col:
                    path = os.path.join(PLOTS_DIR, f"cm_{slug(mname)}.png")
                    if os.path.exists(path):
                        st.image(path, caption=mname, use_container_width=True)
                    else:
                        st.warning(f"No plot for {mname}. Run the pipeline first.")

    with tab2:
        st.markdown('<div class="section-title">ROC Curves (One-vs-Rest) — All Models</div>', unsafe_allow_html=True)
        rows = [models_list]
        for row_models in rows:
            cols = st.columns(len(row_models))
            for col, mname in zip(cols, row_models):
                with col:
                    path = os.path.join(PLOTS_DIR, f"roc_{slug(mname)}.png")
                    if os.path.exists(path):
                        st.image(path, caption=mname, use_container_width=True)
                    else:
                        st.warning(f"No ROC plot for {mname}. Run the pipeline first.")

    with tab3:
        st.markdown('<div class="section-title">Model Comparison Charts</div>', unsafe_allow_html=True)
        comparison_plots = [
            ("comparison_accuracy.png",        "Accuracy Comparison"),
            ("comparison_training_time.png",   "Training Time Comparison"),
            ("comparison_accuracy_vs_f1.png",  "Accuracy vs Macro F1"),
        ]
        for fname, title in comparison_plots:
            path = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
                st.markdown("")
            else:
                st.warning(f"{title} not found. Run the pipeline first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚀  Run Pipeline":
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Run Full Pipeline</h1>
        <p>Train all 5 models from scratch and regenerate all plots and saved models</p>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_action = st.columns([1.4, 1])

    with col_info:
        st.markdown('<div class="section-title">What This Does</div>', unsafe_allow_html=True)
        st.markdown("""
        Clicking **Run Full Pipeline** will execute `run.py` end-to-end:

        1. 📥 **Fetch** the Cleveland dataset from UCI ML Repository
        2. 🔧 **Preprocess** — impute missing values, scale features, stratified split
        3. 🚀 **Train** all 5 models (Logistic Regression, Random Forest, SVM, KNN, XGBoost)
        4. 📊 **Evaluate** — confusion matrices, ROC curves per model
        5. 📈 **Compare** — accuracy, F1, and training time charts
        6. 💾 **Save** all models to `models/` and all plots to `plots/`

        > ⚠️ This may take **1–3 minutes** depending on your machine.
        """)

        st.markdown('<div class="section-title">Saved Outputs</div>', unsafe_allow_html=True)
        col_m, col_p = st.columns(2)
        with col_m:
            st.markdown("**Models (`models/`)**")
            model_files_list = ["logistic_regression.pkl", "svm.pkl", "knn.pkl", "best_model.pkl"]
            for f in model_files_list:
                path = os.path.join(MODELS_DIR, f)
                icon = "✅" if os.path.exists(path) else "❌"
                st.markdown(f"{icon} `{f}`")
        with col_p:
            st.markdown("**Plots (`plots/`)**")
            plot_files_list = os.listdir(PLOTS_DIR) if os.path.exists(PLOTS_DIR) else []
            if plot_files_list:
                for f in sorted(plot_files_list):
                    st.markdown(f"✅ `{f}`")
            else:
                st.markdown("❌ No plots yet")

    # ── Plot Image Gallery (full width below the two columns) ─────────────────
    if os.path.exists(PLOTS_DIR):
        plot_files_list = sorted(os.listdir(PLOTS_DIR))
        if plot_files_list:
            st.markdown('<div class="section-title">Plot Gallery</div>', unsafe_allow_html=True)

            groups = {
                "🔲 Confusion Matrices": [f for f in plot_files_list if f.startswith("cm_")],
                "📈 ROC Curves":         [f for f in plot_files_list if f.startswith("roc_")],
                "⚖️ Comparison Charts":  [f for f in plot_files_list if f.startswith("comparison_")],
            }

            for group_title, files in groups.items():
                if not files:
                    continue
                st.markdown(f"**{group_title}**")
                cols = st.columns(min(len(files), 3))
                for col, fname in zip(cols, files):
                    with col:
                        fpath = os.path.join(PLOTS_DIR, fname)
                        caption = fname.replace(".png", "").replace("_", " ").title()
                        st.image(fpath, caption=caption, use_container_width=True)
                st.markdown("")


    with col_action:
        st.markdown('<div class="section-title">Launch Pipeline</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(239,83,80,0.08); border:1px solid rgba(239,83,80,0.25);
             border-radius:12px; padding:1.2rem; margin-bottom:1rem;'>
            <b style='color:#ef5350;'>⚠️ Note</b><br>
            <span style='color:#bbb; font-size:0.9rem;'>Training downloads data from UCI and may take several minutes.
            Keep this tab open until complete.</span>
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

        if run_btn:
            log_area = st.empty()
            progress_bar = st.progress(0)
            log_lines = []

            run_script = os.path.join(BASE_DIR, "run.py")
            python_exe = sys.executable

            with st.spinner("Pipeline running…"):
                # Force UTF-8 in the child process so emoji in run.py
                # don't crash on Windows with the default cp1252 encoding.
                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8"
                child_env["PYTHONUTF8"] = "1"

                proc = subprocess.Popen(
                    [python_exe, run_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=BASE_DIR,
                    env=child_env,
                )

                step_keywords = ["Fetching", "Preprocessing", "Training", "Evaluating", "Comparing", "Pipeline Complete"]
                progress_val = 0

                for line in proc.stdout:
                    line = line.rstrip()
                    log_lines.append(line)
                    log_area.code("\n".join(log_lines[-60:]), language="bash")

                    for i, kw in enumerate(step_keywords):
                        if kw.lower() in line.lower():
                            progress_val = min(int((i + 1) / len(step_keywords) * 100), 100)
                            progress_bar.progress(progress_val)

                proc.wait()
                progress_bar.progress(100)

            if proc.returncode == 0:
                st.success("✅ Pipeline completed successfully! Switch to **Model Results** to see the updated plots.")
                st.balloons()
            else:
                st.error("❌ Pipeline failed. Check the log output above for details.")
