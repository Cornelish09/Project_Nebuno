import pandas as pd, numpy as np, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA = Path("data/adult.csv")
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
# strip spasi di semua string
obj_cols = df.select_dtypes(include="object").columns.tolist()
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()

df = df.replace("?", np.nan)

# harmonisasi nama umum
if "fnlwgt" in df.columns and "final_weight" not in df.columns:
    df["final_weight"] = df["fnlwgt"]
if "education_num" in df.columns and "educationnum" not in df.columns:
    df["educationnum"] = df["education_num"]

# target
assert "income" in df.columns, "Kolom target 'income' tidak ada."
y = (df["income"].astype(str).str.contains(">50", case=False)).astype(int)

# gender manual (Male=1, Female=0) robust
if "gender" not in df.columns:
    assert "sex" in df.columns, "Kolom 'sex' tidak ada untuk membuat 'gender'."
    sex_clean = df["sex"].astype(str).str.strip().str.lower()
    df["gender"] = sex_clean.map({"male": 1, "female": 0}).astype("Int64")

num_cols = [c for c in ["age","final_weight","educationnum","capital_gain","capital_loss","hours_per_week"] if c in df.columns]
cat_cols = [c for c in ["workclass","marital_status","occupation","relationship","race"] if c in df.columns]
use_cols = num_cols + ["gender"] + cat_cols
missing = [c for c in use_cols if c not in df.columns]
if missing:
    raise ValueError(f"Kolom hilang di dataset: {missing}")

X = df[use_cols].copy()

# imputasi numerik
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())

# imputasi gender
if X["gender"].isna().any():
    mode_g = X["gender"].mode(dropna=True)
    X["gender"] = X["gender"].fillna(int(mode_g.iloc[0]) if not mode_g.empty else 0)
X["gender"] = pd.to_numeric(X["gender"], errors="coerce").fillna(0).astype(int)

# imputasi kategorikal
for c in cat_cols:
    if X[c].isna().any():
        mode_c = X[c].mode(dropna=True)
        X[c] = X[c].fillna(mode_c.iloc[0] if not mode_c.empty else "Unknown")

# SCALING numerik (fit & simpan)
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
joblib.dump(scaler, ART / "scaler.joblib")

# one-hot sesuai notebook (tanpa drop_first)
X_dum = pd.get_dummies(X, columns=cat_cols, drop_first=False)

# pastikan semua numerik
for c in X_dum.columns:
    if X_dum[c].dtype == "object":
        X_dum[c] = pd.to_numeric(X_dum[c], errors="coerce")
X_dum = X_dum.fillna(0)

# simpan artefak skema
with open(ART / "columns.json", "w", encoding="utf-8") as f:
    json.dump(X_dum.columns.tolist(), f, ensure_ascii=False, indent=2)
with open(ART / "schema.json", "w", encoding="utf-8") as f:
    json.dump({"num_cols": num_cols, "cat_cols": cat_cols}, f, ensure_ascii=False, indent=2)

# split & train
X_tr, X_te, y_tr, y_te = train_test_split(X_dum, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

# evaluasi & simpan
y_pr = clf.predict(X_te)
print("Accuracy:", round(accuracy_score(y_te, y_pr), 4))
print(classification_report(y_te, y_pr))
joblib.dump(clf, MODELS / "manual_model.joblib")
print("Saved:", MODELS / "manual_model.joblib", ART / "columns.json", ART / "schema.json", ART / "scaler.joblib")

# confussion matrix
print("Accuracy:", round(accuracy_score(y_te, y_pr), 4))
print(classification_report(y_te, y_pr))

cm = confusion_matrix(y_te, y_pr, labels=[0, 1])
print("Confusion matrix (rows=true, cols=pred):\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
fig, ax = plt.subplots(figsize=(4,4))
disp.plot(ax=ax, values_format="d", colorbar=False)
plt.tight_layout()
plt.savefig(ART / "confusion_matrix.png", dpi=150)
plt.close(fig)
print("Saved:", ART / "confusion_matrix.png")
