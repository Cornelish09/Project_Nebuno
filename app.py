import streamlit as st, pandas as pd, numpy as np, joblib, json, requests
from pathlib import Path

st.set_page_config(page_title="Income Prediction (Notebook-style, Scaled)", page_icon="💼")

GRAD_CSS = """
<style>
:root{
  --grad: linear-gradient(90deg,#FF6BB5 0%, #FF3F8E 22%, #B44CFF 55%, #6A5CFF 78%, #4AA8FF 100%);
  --panel:#161B22; --border:#263042;
}
.block-container{padding-top:1.2rem;}
h1,h2,h3{
  font-weight:800; letter-spacing:.3px;
  background:var(--grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.stButton>button{
  background:var(--grad); color:white; border:0; border-radius:12px;
  padding:.65rem 1rem; font-weight:600;
}
.stButton>button:hover{filter:brightness(1.05); transform:translateY(-1px);}
div[data-testid="stMetric"]{
  background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px;
}
.sidebar .sidebar-content, section[data-testid="stSidebar"]{
  background:var(--panel);
}
hr{border:none; height:1px; background:linear-gradient(90deg,transparent,#334155,transparent);}
</style>
"""
st.markdown(GRAD_CSS, unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "manual_model.joblib"
RELEASE_URL = "https://github.com/Cornelish09/Project_Nebuno/releases/download/v1/manual_model.joblib"

@st.cache_resource
def load_artifacts():
    # Unduh model sekali (saat cold start), lalu cache
    MODEL_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        with requests.get(RELEASE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):  # 1 MB
                    if chunk:
                        f.write(chunk)

    model = joblib.load(MODEL_PATH)

    # baca artifacts lain dari repo (pastikan file2 ini ada)
    trained_cols = json.loads((BASE / "artifacts" / "columns.json").read_text(encoding="utf-8"))
    schema       = json.loads((BASE / "artifacts" / "schema.json").read_text(encoding="utf-8"))
    scaler       = joblib.load(BASE / "artifacts" / "scaler.joblib")

    return model, trained_cols, schema, scaler


model, trained_cols, schema, scaler = load_artifacts()
num_cols, cat_cols = schema["num_cols"], schema["cat_cols"]

st.header("Income Prediction", divider=True)

left, right = st.columns(2)
with st.form("predict_form"):
    with left:
        age = st.number_input("Age", 0, 120, 35)
        fnlwgt = st.number_input("Final weight (fnlwgt)", 0, 1_000_000, 200_000)
        education_num = st.number_input("Education-num", 0, 20, 13)
        hours_per_week = st.number_input("Hours per week", 1, 99, 40)
        sex = st.selectbox("Sex", ["Male","Female"])
        race = st.selectbox("Race", ["White","Asian-Pac-Islander","Amer-Indian-Eskimo","Other","Black"])
    with right:
        workclass = st.selectbox("Workclass", ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov","Without-pay","Never-worked"])
        marital_status = st.selectbox("Marital status", ["Married-civ-spouse","Divorced","Never-married","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"])
        occupation = st.selectbox("Occupation", ["Tech-support","Craft-repair","Other-service","Sales","Exec-managerial","Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"])
        relationship = st.selectbox("Relationship", ["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"])
        capital_gain = st.number_input("Capital gain", 0, 100_000, 0)
        capital_loss = st.number_input("Capital loss", 0, 100_000, 0)
    submitted = st.form_submit_button("Predict", use_container_width=True)

def build_row():
    base = {
        "age": age,
        "final_weight": fnlwgt,
        "educationnum": education_num,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "gender": 1 if sex == "Male" else 0,
        "workclass": workclass,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
    }
    return pd.DataFrame([base])

if submitted:
    X = build_row()

    # imputasi numerik (aman meski jarang NaN dari form)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # imputasi gender (fallback ke 0)
    if "gender" in X.columns and X["gender"].isna().any():
        X["gender"] = X["gender"].fillna(0)
    X["gender"] = pd.to_numeric(X["gender"], errors="coerce").fillna(0).astype(int)

    # imputasi kategorikal
    for c in cat_cols:
        if X[c].isna().any():
            mode_val = X[c].mode().iloc[0] if not X[c].mode().empty else X[c].iloc[0]
            X[c] = X[c].fillna(mode_val)

    # SCALING numerik (pakai scaler hasil training)
    X[num_cols] = scaler.transform(X[num_cols])

    # get_dummies dan reindex ke skema training
    X_dum = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_dum = X_dum.reindex(columns=trained_cols, fill_value=0)

    y = model.predict(X_dum)[0]
    proba = model.predict_proba(X_dum)[0][1] if hasattr(model, "predict_proba") else None
    label = ">50K" if y == 1 else "<=50K"
t1, t2 = st.tabs(["Result", "Details"])
with t1:
    st.success(f"Prediksi: {label}")
    if proba is not None:
        st.write(f"Prob >50K: {proba:.3f}")
with t2:
    st.write("**Numerical features (scaled):**")
    st.dataframe(X[num_cols], use_container_width=True)


