import os, base64, io
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

from log_pipeline import generate_batch

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "data", "models",    "xgb_temporal.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "feature_df.parquet")

app = Flask(__name__)

# ── load model ─────────────────────────────────────────────────────────────────
print("Loading model …")
model = joblib.load(MODEL_PATH)

# FEATURE_COLS is the single source of truth for what goes into the model.
# Since the model was retrained without total_events, it simply will not
# appear in model.feature_names_in_ — no special-casing needed beyond making
# sure we never force it back in.
FEATURE_COLS = list(model.feature_names_in_)
assert "total_events" not in FEATURE_COLS, (
    "total_events should NOT be in the retrained model's feature list. "
    "Re-check xgb_temporal.joblib."
)

# ── load feature dataset ───────────────────────────────────────────────────────
print("Loading feature dataset …")
feature_df = pd.read_parquet(DATA_PATH).copy()

# normalise 'day': parquet may store it as Timestamp; convert to plain string
# so int()/JSON serialisation never fails anywhere in the app
if pd.api.types.is_datetime64_any_dtype(feature_df["day"]):
    feature_df["day"] = feature_df["day"].dt.strftime("%Y-%m-%d")
else:
    feature_df["day"] = feature_df["day"].astype(str)

feature_df = feature_df.sort_values(["day", "user"]).reset_index(drop=True)

# ── display dataframe (total_events fully excluded — never read or shown) ────
df_display = feature_df[["user", "day", "is_insider"]].copy()

# raw activity columns shown in the Risk Overview table
_DISPLAY_RAW_COLS = [
    "file_count", "email_count", "http_count",
    "device_count", "logon_count", "first_hour", "last_hour",
    "unique_pcs", "activity_type_count",
]
for col in _DISPLAY_RAW_COLS:
    if col in feature_df.columns:
        df_display[col] = feature_df[col].values

_leak_cols     = [
    "after_hours_events", "user_mean_after", "user_std_after",
    "after_hours_events_dev", "after_hours_ratio",
]
_mean_std_cols = [c for c in feature_df.columns
                  if c.startswith("user_mean_") or c.startswith("user_std_")]
_id_cols       = ["user", "day", "employee_name", "email", "projects"]
_removed_cols  = ["total_events"]          # ← explicitly excluded from model input
_drop_set      = set(_leak_cols + _mean_std_cols + _id_cols + _removed_cols)

df_model = feature_df.drop(
    columns=[c for c in _drop_set if c in feature_df.columns]
).copy()

# pop target
y_all = df_model.pop("is_insider").astype(int)

# one-hot encode categoricals — same as training
_cat_cols = [c for c in ["role", "functional_unit", "department", "team", "supervisor"]
             if c in df_model.columns]
df_model = pd.get_dummies(df_model, columns=_cat_cols, drop_first=False)

# align columns to exactly what the model expects (FEATURE_COLS, no total_events)
for col in FEATURE_COLS:
    if col not in df_model.columns:
        df_model[col] = 0          # missing dummy columns → 0
df_model = df_model[FEATURE_COLS]  # ensure correct order, drops anything extra

# cast everything to float32 — SHAP TreeExplainer requires numeric types;
# pd.get_dummies on pandas >= 2.0 produces bool columns which cause errors.
X_all = df_model.fillna(0).astype("float32")

# ── pre-compute predictions & risk scores ─────────────────────────────────────
print("Running inference on full dataset …")
df_display["risk_score"]  = model.predict_proba(X_all)[:, 1]
df_display["prediction"]  = model.predict(X_all)
df_display["user_day_id"] = df_display.index

df = df_display.copy()

# ── SHAP explainer (TreeExplainer, fast for XGBoost) ──────────────────────────
print("Initialising SHAP explainer …")
explainer = shap.TreeExplainer(model)

# ── helper: encode matplotlib figure → base64 png ─────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── predefined scenarios ───────────────────────────────────────────────────────
SCENARIOS = {
    "normal_activity": {
        "label": "Normal Activity",
        "description": "Inactive day — no file/device events, typical email and HTTP volume.",
        "values": {
            "file_count": 0,  "email_count": 9,  "device_count": 0,
            "http_count": 95, "logon_count": 2,  "unique_pcs": 0,
            "first_hour": 0,  "last_hour": 0,    "activity_type_count": 2,
        }
    },
    "data_exfiltration": {
        "label": "Data Exfiltration",
        "description": "Unusually high file + device events (USB copy) late at night, accessed from multiple PCs.",
        "values": {
            "file_count": 20, "email_count": 2,  "device_count": 15,
            "http_count": 5,  "logon_count": 3,  "unique_pcs": 2,
            "first_hour": 21, "last_hour": 22,   "activity_type_count": 3,
        }
    },
    "email_exfiltration": {
        "label": "Email Exfiltration",
        "description": "Email volume far above p95, suggesting bulk data emailing.",
        "values": {
            "file_count": 5,  "email_count": 36, "device_count": 0,
            "http_count": 80, "logon_count": 2,  "unique_pcs": 1,
            "first_hour": 0,  "last_hour": 0,    "activity_type_count": 3,
        }
    },
    "off_hours_browsing": {
        "label": "Off-Hours Web Browsing",
        "description": "HTTP count near dataset max with late first/last activity hour.",
        "values": {
            "file_count": 0,   "email_count": 1, "device_count": 0,
            "http_count": 300, "logon_count": 1, "unique_pcs": 1,
            "first_hour": 20,  "last_hour": 22,  "activity_type_count": 2,
        }
    },
}


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── 1. Risk Overview ───────────────────────────────────────────────────────────
@app.route("/api/risk_overview")
def api_risk_overview():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    min_risk = float(request.args.get("min_risk", 0.0))
    search   = request.args.get("search", "").strip().lower()

    filtered = df[df["risk_score"] >= min_risk].copy()
    if search:
        filtered = filtered[
            filtered["user"].astype(str).str.lower().str.contains(search)
        ]
    filtered = filtered.sort_values("risk_score", ascending=False)

    total   = len(filtered)
    start   = (page - 1) * per_page
    end     = start + per_page
    records = filtered.iloc[start:end]

    rows = []
    for _, r in records.iterrows():
        rows.append({
            "user_day_id" : int(r["user_day_id"]),
            "user"        : str(r["user"]),
            "day"         : str(r["day"]),
            "risk_score"  : round(float(r["risk_score"]), 4),
            "prediction"  : int(r["prediction"]),
            "is_insider"  : int(r["is_insider"]),
            "unique_pcs"  : int(r.get("unique_pcs",   0)),
            "file_count"  : int(r.get("file_count",   0)),
            "email_count" : int(r.get("email_count",  0)),
            "http_count"  : int(r.get("http_count",   0)),
            "device_count": int(r.get("device_count", 0)),
        })

    return jsonify({"total": total, "page": page, "per_page": per_page, "rows": rows})


# ── 2. User-Day Detail + SHAP force plot ──────────────────────────────────────
@app.route("/api/user_day/<int:uid>")
def api_user_day(uid):
    try:
        return _api_user_day_inner(uid)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


def _api_user_day_inner(uid):
    row = df[df["user_day_id"] == uid]
    if row.empty:
        return jsonify({"error": "Not found"}), 404

    r = row.iloc[0]

    # uid == positional index into X_all (both were built from the same
    # sorted feature_df, so row position is the same)
    x  = X_all.iloc[[uid]]          # single-row DataFrame, keeps column names
    sv = explainer.shap_values(x)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]

    # top contributing features for this record (FEATURE_COLS has no total_events)
    feat_contribs = sorted(
        zip(FEATURE_COLS, sv.tolist()),
        key=lambda t: abs(t[1]), reverse=True
    )[:15]

    # force plot as image
    fig, ax = plt.subplots(figsize=(12, 3))
    names  = [f[0] for f in feat_contribs]
    vals   = [f[1] for f in feat_contribs]
    colors = ["#e05252" if v > 0 else "#5282e0" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"Local SHAP — User {r['user']}  Day {r['day']}")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    force_b64 = fig_to_b64(fig)

    # raw feature values for the detail panel (only model-input features)
    feat_vals = {}
    for c in FEATURE_COLS:
        v = x[c].iloc[0]
        try:
            feat_vals[c] = float(v)
        except (TypeError, ValueError):
            feat_vals[c] = str(v)

    return jsonify({
        "user"             : str(r["user"]),
        "day"              : str(r["day"]),
        "risk_score"       : round(float(r["risk_score"]), 4),
        "prediction"       : int(r["prediction"]),
        "is_insider"       : int(r["is_insider"]),
        "feat_contribs"    : feat_contribs,
        "feat_vals"        : feat_vals,
        "force_plot"       : force_b64,
    })


# ── 3. Temporal Trend ─────────────────────────────────────────────────────────
@app.route("/api/temporal_trend")
def api_temporal_trend():
    daily = (df.groupby("day")
               .agg(avg_risk=("risk_score","mean"),
                    flagged=("prediction","sum"),
                    total=("prediction","count"),
                    insider_count=("is_insider","sum"))
               .reset_index()
               .sort_values("day"))

    return jsonify({
        "days"          : daily["day"].tolist(),
        "avg_risk"      : daily["avg_risk"].round(4).tolist(),
        "flagged"       : daily["flagged"].tolist(),
        "total"         : daily["total"].tolist(),
        "insider_count" : daily["insider_count"].tolist(),
    })


# ── 4. Global SHAP summary (top-15 bar) ───────────────────────────────────────
@app.route("/api/shap_global")
def api_shap_global():
    sample = X_all.sample(min(500, len(X_all)), random_state=42)
    sv     = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]

    mean_abs = np.abs(sv).mean(axis=0)
    pairs    = sorted(zip(FEATURE_COLS, mean_abs.tolist()),
                      key=lambda t: t[1], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [p[0] for p in pairs][::-1]
    vals  = [p[1] for p in pairs][::-1]
    ax.barh(names, vals, color="#4f6ef7")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 SHAP Feature Importances — XGBoost (CERT r6.2)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    img_b64 = fig_to_b64(fig)

    return jsonify({"image": img_b64, "features": names[::-1], "values": vals[::-1]})


# ── 5. Scenario Testing ───────────────────────────────────────────────────────
# ── shared helper: score an arbitrary engineered feature dict ────────────────
def score_feature_row(engineered: dict, medians: pd.Series = None):
    """
    Take a dict of engineered behavioural features (e.g. from
    log_pipeline.parse_and_engineer or a scenario definition), fill in any
    features the model expects but weren't supplied using dataset medians,
    align to FEATURE_COLS, run the model + SHAP, and return a result dict.

    `engineered` keys may include non-model fields (user, day, n_log_lines)
    — these are ignored when building the model input row.
    """
    if medians is None:
        medians = X_all.median()

    row_dict = medians.to_dict()
    for feat, val in engineered.items():
        if feat in row_dict:
            row_dict[feat] = val
        dev_key = feat + "_dev"
        if dev_key in row_dict:
            row_dict[dev_key] = val - medians.get(feat, val)

    x_row = pd.DataFrame([row_dict])[FEATURE_COLS].fillna(0).astype("float32")
    prob  = float(model.predict_proba(x_row)[0, 1])
    pred  = int(model.predict(x_row)[0])

    sv = explainer.shap_values(x_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]

    contribs = sorted(zip(FEATURE_COLS, sv.tolist()),
                      key=lambda t: abs(t[1]), reverse=True)[:12]

    return {
        "risk_score": round(prob, 4),
        "prediction": pred,
        "contribs"  : contribs,
        "x_row"     : x_row,
    }


def _shap_waterfall_fig(contribs, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    names  = [c[0] for c in contribs][::-1]
    vals   = [c[1] for c in contribs][::-1]
    colors = ["#e05252" if v > 0 else "#5282e0" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(title)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify(list(SCENARIOS.keys()))


@app.route("/api/scenario/<key>")
def api_scenario(key):
    if key not in SCENARIOS:
        return jsonify({"error": "Unknown scenario"}), 404

    sc     = SCENARIOS[key]
    result = score_feature_row(sc["values"])

    fig = _shap_waterfall_fig(
        result["contribs"],
        f"Scenario: {sc['label']}  —  Risk score: {result['risk_score']:.3f}"
    )
    img_b64 = fig_to_b64(fig)

    return jsonify({
        "key"        : key,
        "label"      : sc["label"],
        "description": sc["description"],
        "risk_score" : result["risk_score"],
        "prediction" : result["prediction"],
        "contribs"   : result["contribs"],
        "image"      : img_b64,
    })


# ── 6. Log Simulation — fully automatic batch pipeline ───────────────────────
#
#   Backend does everything in one call:
#     generate random logs for N users → parse each → engineer features
#     → score through XGBoost → run SHAP → return ranked results
#
#   In-memory cache so the User-Day-style drill-down (SHAP force plot) can
#   look up a specific generated record by id without regenerating it.
_LOGSIM_CACHE = {"records": []}


def _run_logsim_batch(n: int = 30, seed: int = None):
    batch = generate_batch(n=n, seed=seed)
    medians = X_all.median()

    records = []
    for i, item in enumerate(batch):
        engineered   = item["engineered"]
        meta_fields  = {"user", "day", "n_log_lines"}
        model_inputs = {k: v for k, v in engineered.items() if k not in meta_fields}

        result = score_feature_row(model_inputs, medians=medians)

        records.append({
            "sim_id"        : i,
            "user"          : item["user"],
            "day"           : item["day"],
            "scenario_key"  : item["scenario_key"],
            "scenario_label": item["scenario_label"],
            "risk_score"    : result["risk_score"],
            "prediction"    : result["prediction"],
            "contribs"      : result["contribs"],
            "engineered"    : engineered,
            "csv_text"      : item["csv_text"],
            "readable_text" : item["readable_text"],
        })

    records.sort(key=lambda r: r["risk_score"], reverse=True)
    _LOGSIM_CACHE["records"] = records
    return records


@app.route("/api/log_sim/run")
def api_log_sim_run():
    """
    Fully automatic pipeline trigger.
    ?n=<count>  (default 30, clamped 5-50)

    Generates N random user-day logs, parses + engineers features,
    scores through the model, and returns a ranked summary table
    (each record's full SHAP image is fetched lazily via drill-down).
    """
    n = int(request.args.get("n", 30))
    n = max(5, min(50, n))

    try:
        records = _run_logsim_batch(n=n)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    rows = []
    for r in records:
        rows.append({
            "sim_id"        : r["sim_id"],
            "user"          : r["user"],
            "day"           : r["day"],
            "scenario_label": r["scenario_label"],
            "risk_score"    : r["risk_score"],
            "prediction"    : r["prediction"],
            "file_count"    : r["engineered"].get("file_count", 0),
            "email_count"   : r["engineered"].get("email_count", 0),
            "http_count"    : r["engineered"].get("http_count", 0),
            "device_count"  : r["engineered"].get("device_count", 0),
            "unique_pcs"    : r["engineered"].get("unique_pcs", 0),
        })

    flagged_count  = sum(1 for r in records if r["prediction"] == 1)
    avg_risk       = sum(r["risk_score"] for r in records) / len(records) if records else 0

    return jsonify({
        "total"        : len(records),
        "flagged_count": flagged_count,
        "avg_risk"     : round(avg_risk, 4),
        "rows"         : rows,
    })


@app.route("/api/log_sim/detail/<int:sim_id>")
def api_log_sim_detail(sim_id):
    """
    Drill-down into a single auto-generated record from the last batch run:
    returns the raw log, readable translation, engineered features, and
    a rendered SHAP force-plot image for that specific record.
    """
    records = _LOGSIM_CACHE["records"]
    match = next((r for r in records if r["sim_id"] == sim_id), None)
    if match is None:
        return jsonify({"error": "Record not found. Re-run the simulation."}), 404

    fig = _shap_waterfall_fig(
        match["contribs"],
        f"Log Simulation — User {match['user']}  Day {match['day']}  "
        f"({match['scenario_label']})  —  Risk score: {match['risk_score']:.3f}"
    )
    img_b64 = fig_to_b64(fig)

    return jsonify({
        "sim_id"         : match["sim_id"],
        "user"           : match["user"],
        "day"            : match["day"],
        "scenario_label" : match["scenario_label"],
        "risk_score"     : match["risk_score"],
        "prediction"     : match["prediction"],
        "engineered"     : match["engineered"],
        "csv_text"       : match["csv_text"],
        "readable_text"  : match["readable_text"],
        "contribs"       : match["contribs"],
        "image"          : img_b64,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)