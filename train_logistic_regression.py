"""
Script retraining model Student Performance menggunakan Logistic Regression.
Jalankan dari folder utama project:

    python train_logistic_regression.py

Output akan otomatis disimpan ke folder:
    Artifak Model Student Performance/
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "StudentPerformanceFactors.csv"
ARTIFACT_DIR = BASE_DIR / "Artifak Model Student Performance"
ARTIFACT_DIR.mkdir(exist_ok=True)


def kategorikan_performa(score: float) -> str:
    """Membuat target kategori performa dari Exam_Score."""
    if score >= 70:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


# 1. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# 2. Buat target klasifikasi
if "Exam_Score" not in df.columns:
    raise ValueError("Kolom Exam_Score tidak ditemukan di dataset.")

df["Performance_Category"] = df["Exam_Score"].apply(kategorikan_performa)

# 3. Fitur yang digunakan pada project Streamlit
# Catatan: daftar ini disesuaikan dengan struktur artefak project saat ini.
cat_features = [
    "Access_to_Resources",
    "Parental_Involvement",
    "Family_Income",
    "Motivation_Level",
    "Parental_Education_Level",
    "Learning_Disabilities",
    "Peer_Influence",
    "Teacher_Quality",
    "Distance_from_Home",
    "Internet_Access",
    "Extracurricular_Activities",
]

num_features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

required_columns = cat_features + num_features + ["Performance_Category"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing_columns}")

# 4. Data preparation: OHE fitur kategorikal + Label Encoding target
df_model = df[cat_features + num_features + ["Performance_Category"]].copy()
X_encoded = pd.get_dummies(df_model.drop(columns=["Performance_Category"]))

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_model["Performance_Category"])

# 5. Split data 80:20 sesuai skenario terbaik notebook
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

# 6. Training Logistic Regression
model = LogisticRegression(
    random_state=42,
    max_iter=3000,
    solver="lbfgs",
)
model.fit(X_train, y_train)

# 7. Evaluasi singkat
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 70)
print("HASIL TRAINING MODEL LOGISTIC REGRESSION")
print("=" * 70)
print(f"Jumlah data              : {len(df_model)}")
print(f"Jumlah fitur setelah OHE : {X_encoded.shape[1]}")
print(f"Kelas target             : {list(label_encoder.classes_)}")
print(f"Akurasi data uji 80:20   : {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Export artefak ke folder yang dipakai app.py
with open(ARTIFACT_DIR / "model_student_performance.pkl", "wb") as f:
    pickle.dump(model, f)

with open(ARTIFACT_DIR / "feature_columns.pkl", "wb") as f:
    pickle.dump(X_encoded.columns.tolist(), f)

with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nArtefak berhasil disimpan ke:")
print(f"- {ARTIFACT_DIR / 'model_student_performance.pkl'}")
print(f"- {ARTIFACT_DIR / 'feature_columns.pkl'}")
print(f"- {ARTIFACT_DIR / 'label_encoder.pkl'}")
print("=" * 70)
