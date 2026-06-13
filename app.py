"""
app.py — Valorant Match Predictor Flask Backend
"""

import json, os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model_metadata.json")
DATA_PATH  = os.path.join(BASE_DIR, "valorant_match_data.csv")

# ── Load model & metadata ──
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl tidak ditemukan. Jalankan 'python modeling.py' dulu.")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("model_metadata.json tidak ditemukan.")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    print(f"[INFO] Model loaded     : {metadata.get('model_type','?')}")
    print(f"[INFO] Model accuracy   : {metadata.get('metrics',{}).get('accuracy','?')}")
    return model, metadata

model, metadata = load_artifacts()

# ── Compute dashboard analytics from CSV ──
def compute_analytics():
    if not os.path.exists(DATA_PATH):
        return {}
    df = pd.read_csv(DATA_PATH)

    # Win rate by Map
    map_wr = (
        df.groupby("Map")["TeamWin"]
        .agg(["mean","count"])
        .reset_index()
        .rename(columns={"mean":"win_rate","count":"total"})
        .sort_values("win_rate", ascending=False)
    )

    # Win rate by Rank
    rank_wr = (
        df.groupby("AverageRank")["TeamWin"]
        .agg(["mean","count"])
        .reset_index()
        .rename(columns={"mean":"win_rate","count":"total"})
        .sort_values("win_rate", ascending=False)
    )

    # KD Ratio bins
    df["kd_bin"] = pd.cut(df["KD_Ratio"], bins=10)
    kd_dist = (
        df.groupby("kd_bin", observed=True)["TeamWin"]
        .agg(["mean","count"])
        .reset_index()
    )
    kd_dist["label"] = kd_dist["kd_bin"].apply(lambda x: f"{x.left:.1f}")

    # Spike plants impact
    df["plant_bin"] = pd.cut(df["SpikePlants"], bins=5)
    plant_wr = (
        df.groupby("plant_bin", observed=True)["TeamWin"]
        .mean()
        .reset_index()
    )
    plant_wr["label"] = plant_wr["plant_bin"].apply(lambda x: f"{x.left:.0f}–{x.right:.0f}")

    # First Bloods impact
    df["fb_bin"] = pd.cut(df["FirstBloods"], bins=6)
    fb_wr = (
        df.groupby("fb_bin", observed=True)["TeamWin"]
        .mean()
        .reset_index()
    )
    fb_wr["label"] = fb_wr["fb_bin"].apply(lambda x: f"{x.left:.0f}–{x.right:.0f}")

    # Overall stats
    total_matches = len(df)
    overall_wr    = round(df["TeamWin"].mean() * 100, 1)
    avg_kd        = round(df["KD_Ratio"].mean(), 2)
    avg_kills     = round(df["Kills"].mean(), 1)
    avg_fb        = round(df["FirstBloods"].mean(), 1)
    avg_plants    = round(df["SpikePlants"].mean(), 1)

    return {
        "map_labels":   map_wr["Map"].tolist(),
        "map_wr":       [round(x*100,1) for x in map_wr["win_rate"].tolist()],
        "map_total":    map_wr["total"].tolist(),
        "rank_labels":  rank_wr["AverageRank"].tolist(),
        "rank_wr":      [round(x*100,1) for x in rank_wr["win_rate"].tolist()],
        "kd_labels":    kd_dist["label"].tolist(),
        "kd_wr":        [round(x*100,1) for x in kd_dist["mean"].tolist()],
        "kd_count":     kd_dist["count"].tolist(),
        "plant_labels": plant_wr["label"].tolist(),
        "plant_wr":     [round(x*100,1) for x in plant_wr["TeamWin"].tolist()],
        "fb_labels":    fb_wr["label"].tolist(),
        "fb_wr":        [round(x*100,1) for x in fb_wr["TeamWin"].tolist()],
        "total_matches": total_matches,
        "overall_wr":   overall_wr,
        "avg_kd":       avg_kd,
        "avg_kills":    avg_kills,
        "avg_fb":       avg_fb,
        "avg_plants":   avg_plants,
    }

analytics = compute_analytics()

# ── Helpers ──
def safe_float(val, default=0.0):
    try: return float(val)
    except: return default

# ── Feature vector builder ──
def build_feature_vector(form_data: dict) -> np.ndarray:
    encoder_maps = metadata["encoder_maps"]
    features     = metadata["features"]
    map_enc  = encoder_maps["Map"].get(form_data.get("Map",""), 0)
    rank_enc = encoder_maps["AverageRank"].get(form_data.get("AverageRank",""), 0)
    vector = {
        "Map_enc":     map_enc,
        "Rank_enc":    rank_enc,
        "EconRating":  safe_float(form_data.get("EconRating")),
        "FirstBloods": safe_float(form_data.get("FirstBloods")),
        "SpikePlants": safe_float(form_data.get("SpikePlants")),
        "Kills":       safe_float(form_data.get("Kills")),
        "Deaths":      safe_float(form_data.get("Deaths")),
        "KD_Ratio":    safe_float(form_data.get("KD_Ratio")),
    }
    return np.array([[vector[f] for f in features]])

# ── Routes ──
@app.route("/")
def index():
    map_list  = sorted(metadata["encoder_maps"]["Map"].keys())
    rank_list = sorted(
        metadata["encoder_maps"]["AverageRank"].keys(),
        key=lambda r: metadata["encoder_maps"]["AverageRank"][r]
    )
    return render_template(
        "index.html",
        map_list           = map_list,
        rank_list          = rank_list,
        feature_stats      = metadata.get("feature_stats", {}),
        model_metrics      = metadata.get("metrics", {}),
        feature_importance = metadata.get("feature_importance", []),
        analytics          = analytics,
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Request body kosong."}), 400
        required = ["Map","AverageRank","EconRating","FirstBloods","SpikePlants","Kills","Deaths","KD_Ratio"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Field tidak ada: {missing}"}), 400
        valid_maps  = list(metadata["encoder_maps"]["Map"].keys())
        valid_ranks = list(metadata["encoder_maps"]["AverageRank"].keys())
        if data["Map"] not in valid_maps:
            return jsonify({"error": f"Map tidak valid: {valid_maps}"}), 400
        if data["AverageRank"] not in valid_ranks:
            return jsonify({"error": f"Rank tidak valid: {valid_ranks}"}), 400

        X = build_feature_vector(data)
        prediction = int(model.predict(X)[0])
        proba      = float(model.predict_proba(X)[0][1])
        label      = "VICTORY" if prediction == 1 else "DEFEAT"
        win_pct    = round(proba * 100, 1)
        if win_pct >= 70 or win_pct <= 30:
            confidence = "High"
        elif 55 <= win_pct <= 70 or 30 <= win_pct <= 45:
            confidence = "Medium"
        else:
            confidence = "Low"
        # Build input stats for tactical analysis
        input_stats = {
            "KD_Ratio":    safe_float(data.get("KD_Ratio")),
            "FirstBloods": safe_float(data.get("FirstBloods")),
            "SpikePlants": safe_float(data.get("SpikePlants")),
            "EconRating":  safe_float(data.get("EconRating")),
            "Kills":       safe_float(data.get("Kills")),
            "Deaths":      safe_float(data.get("Deaths")),
        }
        feature_stats = metadata.get("feature_stats", {})
        feature_imp   = metadata.get("feature_importance", [])

        return jsonify({
            "prediction": prediction, "label": label,
            "probability": win_pct, "confidence": confidence,
            "input_stats": input_stats,
            "feature_stats": feature_stats,
            "feature_importance": feature_imp,
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status":"ok","model":metadata.get("model_type","?"),
                    "accuracy":metadata.get("metrics",{}).get("accuracy","?")})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
