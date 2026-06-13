"""
modeling.py
===========
Valorant Competitive Match Outcome Predictor
Skrip training model klasifikasi biner (TeamWin: 1=Victory, 0=Defeat)
menggunakan Gradient Boosting Classifier dengan MLflow + DagsHub tracking.

Jalankan:
    python modeling.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix
)

import mlflow
import mlflow.sklearn
import dagshub

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. KONFIGURASI — Ubah sesuai akun DagsHub
# ─────────────────────────────────────────────
DAGSHUB_USERNAME  = "anndaanhr"    # DagsHub username
DAGSHUB_REPO_NAME = "UASDS"        # Nama repo di DagsHub
EXPERIMENT_NAME   = "modelprediksimatchvalorant"

DATASET_PATH = "valorant_match_data.csv"
MODEL_PATH   = "model.pkl"
META_PATH    = "model_metadata.json"

# ─────────────────────────────────────────────
# 1. INISIALISASI DAGSHUB + MLFLOW
#    Jika DagsHub gagal → otomatis fallback ke MLflow lokal
# ─────────────────────────────────────────────
def init_tracking():
    try:
        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True
        )
        print(f"[INFO] DagsHub MLflow aktif : https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")
    except Exception as e:
        print(f"[WARNING] DagsHub tidak dapat diakses ({e}). Menggunakan MLflow lokal.")
        mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[INFO] MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"[INFO] Experiment           : {EXPERIMENT_NAME}")

# ─────────────────────────────────────────────
# 2. LOAD & VALIDASI DATASET
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' tidak ditemukan. "
            "Pastikan file ada di direktori yang sama dengan modeling.py"
        )
    df = pd.read_csv(path)
    print(f"\n[INFO] Dataset loaded   : {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"[INFO] Columns          : {list(df.columns)}")
    print(f"[INFO] Missing values   :\n{df.isnull().sum()}")
    print(f"[INFO] Target balance   :\n{df['TeamWin'].value_counts()}")
    return df

# ─────────────────────────────────────────────
# 3. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Langkah:
    1. Drop MatchID (ID tidak relevan untuk model)
    2. Encode kolom kategorikal (Map, AverageRank) dengan LabelEncoder
    3. Pilih fitur yang paling informatif untuk prediksi TeamWin
    4. Kembalikan X, y, encoder maps, dan daftar fitur
    """

    df = df.copy()

    # Hapus duplikat & baris kosong
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"\n[INFO] Setelah cleaning : {df.shape[0]} rows")

    # Kolom kategorikal
    rank_order = ["Iron", "Bronze", "Silver", "Gold", "Platinum",
                  "Diamond", "Ascendant", "Immortal", "Radiant"]

    le_map  = LabelEncoder()
    le_rank = LabelEncoder()

    df["Map_enc"]  = le_map.fit_transform(df["Map"])
    df["Rank_enc"] = le_rank.fit_transform(df["AverageRank"])

    # Fitur yang dipilih (8 fitur terbaik berdasarkan domain knowledge Valorant)
    FEATURES = [
        "Map_enc",       # Peta berpengaruh pada strategi
        "Rank_enc",      # Rata-rata rank tim
        "EconRating",    # Efisiensi ekonomi
        "FirstBloods",   # First blood = momentum ronde
        "SpikePlants",   # Spike plant = kontrol objektif
        "Kills",         # Total kill tim
        "Deaths",        # Total kematian tim
        "KD_Ratio"       # Rasio K/D keseluruhan
    ]
    TARGET = "TeamWin"

    X = df[FEATURES]
    y = df[TARGET]

    # Simpan info encoder untuk metadata
    encoder_maps = {
        "Map": {str(cls): int(idx) for idx, cls in enumerate(le_map.classes_)},
        "AverageRank": {str(cls): int(idx) for idx, cls in enumerate(le_rank.classes_)}
    }

    print(f"\n[INFO] Fitur yang digunakan : {FEATURES}")
    print(f"[INFO] Target               : {TARGET}")
    print(f"[INFO] Map encoding         : {encoder_maps['Map']}")
    print(f"[INFO] Rank encoding        : {encoder_maps['AverageRank']}")

    return X, y, FEATURES, encoder_maps

# ─────────────────────────────────────────────
# 4. TRAINING & EVALUASI
# ─────────────────────────────────────────────
def train_and_evaluate(X, y):
    """
    Train Gradient Boosting Classifier dengan hyperparameter tuning ringan.
    Evaluasi menggunakan 80/20 split + 5-fold cross-validation.
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train size : {X_train.shape[0]}")
    print(f"[INFO] Test size  : {X_test.shape[0]}")

    # Hyperparameter model
    params = {
        "n_estimators"     : 200,
        "learning_rate"    : 0.05,
        "max_depth"        : 4,
        "min_samples_split": 10,
        "min_samples_leaf" : 5,
        "subsample"        : 0.8,
        "random_state"     : 42
    }

    model = GradientBoostingClassifier(**params)

    # Cross-validation (5-fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\n[INFO] CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Training final model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Hitung metrik
    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1_score" : f1_score(y_test, y_pred),
        "cv_mean"  : float(cv_scores.mean()),
        "cv_std"   : float(cv_scores.std())
    }

    print("\n" + "=" * 50)
    print("  HASIL EVALUASI MODEL")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Defeat", "Victory"]))
    print("[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, params, metrics, X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def get_feature_importance(model, features: list) -> list:
    importances = model.feature_importances_
    fi = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )
    print("\n[INFO] Feature Importance (Top 8):")
    for name, score in fi:
        bar = "|" * int(score * 50)
        print(f"  {name:18s}: {score:.4f}  {bar}")
    return fi

# ─────────────────────────────────────────────
# 6. SIMPAN MODEL + METADATA
# ─────────────────────────────────────────────
def save_artifacts(model, features, encoder_maps, feature_importance, X, metrics):
    # Simpan model
    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Model disimpan ke : {MODEL_PATH}")

    # Hitung statistik fitur numerik (untuk normalisasi input di app.py)
    numeric_features = ["EconRating", "FirstBloods", "SpikePlants",
                        "Kills", "Deaths", "KD_Ratio"]
    feature_stats = {}
    for col in numeric_features:
        feature_stats[col] = {
            "mean": float(X[col].mean()),
            "min" : float(X[col].min()),
            "max" : float(X[col].max())
        }

    # Buat metadata JSON
    metadata = {
        "model_type"         : "GradientBoostingClassifier",
        "target"             : "TeamWin",
        "target_labels"      : {"0": "Defeat", "1": "Victory"},
        "features"           : features,
        "feature_importance" : [
            {"feature": name, "importance": float(score)}
            for name, score in feature_importance
        ],
        "top_features"       : [fi[0] for fi in feature_importance[:5]],
        "encoder_maps"       : encoder_maps,
        "feature_stats"      : feature_stats,
        "metrics"            : {k: round(v, 4) for k, v in metrics.items()}
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Metadata disimpan ke : {META_PATH}")
    return metadata

# ─────────────────────────────────────────────
# 7. LOG KE MLFLOW (DAGSHUB)
# ─────────────────────────────────────────────
def log_to_mlflow(model, params, metrics, metadata):
    with mlflow.start_run(run_name="GradientBoosting-v1"):
        # Log hyperparameter
        mlflow.log_params(params)

        # Log metrik evaluasi
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model ke MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ValorantMatchPredictor"
        )

        # Log metadata JSON sebagai artefak
        mlflow.log_artifact(META_PATH)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[SUCCESS] MLflow run selesai! Run ID : {run_id}")
        print(f"[INFO] Lihat eksperimen di : https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VALORANT MATCH OUTCOME PREDICTOR — Training Pipeline")
    print("=" * 60)

    # Inisialisasi tracking
    init_tracking()

    # Load dataset
    df = load_data(DATASET_PATH)

    # Preprocessing
    X, y, features, encoder_maps = preprocess(df)

    # Training & evaluasi
    model, params, metrics, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)

    # Feature importance
    feature_importance = get_feature_importance(model, features)

    # Simpan model & metadata
    metadata = save_artifacts(model, features, encoder_maps, feature_importance, X, metrics)

    # Log ke DagsHub MLflow
    log_to_mlflow(model, params, metrics, metadata)

    print("\n" + "=" * 60)
    print("  PIPELINE SELESAI!")
    print(f"  ✓ model.pkl            → siap digunakan di app.py")
    print(f"  ✓ model_metadata.json  → fitur & encoder tersimpan")
    print(f"  ✓ MLflow run           → tercatat di DagsHub")
    print("=" * 60 + "\n")
