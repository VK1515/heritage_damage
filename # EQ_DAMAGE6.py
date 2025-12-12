"""
Heritage Structures — Earthquake Damage Risk Assessment (Enhanced Deployment-ready Streamlit app)
Author: Vrushali Kamalakar (enhanced)
Purpose: Single-file Streamlit application with optional LightGBM, CatBoost, NGBoost, SHAP support.
Run:
    streamlit run heritage_deploy_app.py
Notes:
 - Optional libraries (lightgbm, catboost, ngboost, shap, imbalanced-learn) are imported with safe fallbacks.
 - The app will run with scikit-learn only; optional features are enabled if those packages are installed.
"""

import os
import io
import sys
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# sklearn core
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
import matplotlib.pyplot as plt

# Optional, safe imports
JOBLIB_AVAILABLE = False
LGB_AVAILABLE = False
CAT_AVAILABLE = False
NGBOOST_AVAILABLE = False
SHAP_AVAILABLE = False
IMBLEARN_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CAT_AVAILABLE = True
except Exception:
    CatBoostClassifier = None
    CAT_AVAILABLE = False

try:
    # ngboost provides probabilistic classification/regression
    from ngboost import NGBClassifier

    NGBOOST_AVAILABLE = True
except Exception:
    NGBClassifier = None
    NGBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from imblearn.combine import SMOTETomek

    IMBLEARN_AVAILABLE = True
except Exception:
    SMOTETomek = None
    IMBLEARN_AVAILABLE = False

# Folium mapping
import folium
import streamlit.components.v1 as components

st.set_page_config(page_title="Heritage Damage — Enhanced Deploy", layout="wide")
st.title("Heritage Structures — Earthquake Damage Risk Assessment (Enhanced)")

# ----------------------
# Utility helpers
# ----------------------


def to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    out = BytesIO()
    df.to_excel(out, index=False, engine="openpyxl")
    out.seek(0)
    return out


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Return distance in km (vectorized). Handles arrays or scalars."""
    try:
        lat1, lon1, lat2, lon2 = map(
            np.radians,
            (
                np.array(lat1, dtype=float),
                np.array(lon1, dtype=float),
                float(lat2),
                float(lon2),
            ),
        )
    except Exception:
        try:
            n = len(lat1)
            return np.full(n, np.nan)
        except Exception:
            return np.nan
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def folium_map_from_df(
    df,
    lat_col="location_lat",
    lon_col="location_lon",
    label_col="predicted_damage",
    popup_cols=None,
    start_zoom=6,
):
    if df.empty:
        return "<p>No data to map.</p>"
    center = [float(df[lat_col].mean()), float(df[lon_col].mean())]
    m = folium.Map(location=center, zoom_start=start_zoom)
    color_map = {"minor": "green", "moderate": "orange", "major": "red"}
    for _, r in df.iterrows():
        lab = r.get(label_col, "")
        color = color_map.get(str(lab), "blue")
        if popup_cols:
            popup = "<br>".join([f"<b>{c}</b>: {r.get(c,'')}" for c in popup_cols])
            popup = folium.Popup(popup, max_width=450)
        else:
            popup = folium.Popup(f"{label_col}: {lab}", max_width=250)
        try:
            folium.CircleMarker(
                location=[float(r[lat_col]), float(r[lon_col])],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(m)
        except Exception:
            continue
    return m.get_root().render()


def canonicalize_labels(series: pd.Series) -> pd.Series:
    canonical = {
        "minor": "minor",
        "Minor": "minor",
        "MINOR": "minor",
        "moderate": "moderate",
        "Moderate": "moderate",
        "MODERATE": "moderate",
        "major": "major",
        "Major": "major",
        "MAJOR": "major",
        "Red": "major",
        "Orange": "moderate",
        "Green": "minor",
    }
    return series.map(lambda x: canonical.get(x, x))


def save_artifact(obj, path: str) -> None:
    """Save model artifact with joblib if available, otherwise pickle."""
    if JOBLIB_AVAILABLE:
        joblib.dump(obj, path)
    else:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def load_artifact(path: str):
    if JOBLIB_AVAILABLE:
        return joblib.load(path)
    else:
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)


# ----------------------
# UI: Sidebar - info & file upload
# ----------------------
with st.sidebar:
    st.markdown("### Input files")
    st.write("- `building_data` (CSV/XLSX) — required")
    st.write("- `damage_reports` (CSV/XLSX) — required")
    st.write("- `earthquake_data` (CSV/XLSX) — optional (for epicenter)")
    st.markdown("---")
    st.markdown("### Optional libraries detected")
    st.write(f"LightGBM: {LGB_AVAILABLE}, CatBoost: {CAT_AVAILABLE}")
    st.write(f"NGBoost: {NGBOOST_AVAILABLE}, SHAP: {SHAP_AVAILABLE}")
    st.write(f"SMOTETomek (imbalanced-learn): {IMBLEARN_AVAILABLE}")

st.header("Upload inputs")
col_up1, col_up2, col_up3 = st.columns([3, 3, 2])
with col_up1:
    bld_file = st.file_uploader("Upload building_data (CSV / XLSX)", type=["csv", "xlsx", "xls"])
with col_up2:
    dmg_file = st.file_uploader("Upload damage_reports (CSV / XLSX)", type=["csv", "xlsx", "xls"])
with col_up3:
    eq_file = st.file_uploader("Optional: earthquake_data (CSV / XLSX)", type=["csv", "xlsx", "xls"])

if not bld_file or not dmg_file:
    st.info("Please upload both building_data and damage_reports to proceed.")
    st.stop()


def read_file(uploaded):
    if uploaded is None:
        raise RuntimeError("No file provided.")
    name = uploaded.name.lower()
    content = uploaded.read()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            return pd.read_csv(io.BytesIO(content), encoding="utf-8", errors="replace")
    else:
        return pd.read_excel(io.BytesIO(content))


try:
    building_df = read_file(bld_file)
    damage_df = read_file(dmg_file)
    eq_df = read_file(eq_file) if eq_file else None
except Exception as e:
    st.error(f"Failed to read files: {e}")
    st.stop()

# ----------------------
# Epicenter & distance
# ----------------------
st.header("Epicenter & distance computation")
use_provided_distance = st.checkbox(
    "Use existing distance_to_epicenter if present (from building_data)", value=True
)

if eq_df is not None and not eq_df.empty:
    epi_idx = st.number_input(
        "Epicenter row index (from earthquake_data)",
        min_value=0,
        max_value=max(0, len(eq_df) - 1),
        value=0,
    )
    try:
        epic_lat, epic_lon = None, None
        row = eq_df.iloc[int(epi_idx)]
        for c in ["epicenter_lat", "latitude", "lat", "epi_lat", "epic_lat"]:
            if c in row.index and pd.notna(row[c]):
                epic_lat = float(row[c])
                break
        for c in ["epicenter_lon", "longitude", "lon", "epi_lon", "epic_lon"]:
            if c in row.index and pd.notna(row[c]):
                epic_lon = float(row[c])
                break
    except Exception:
        epic_lat, epic_lon = None, None
else:
    epic_lat = None
    epic_lon = None

if use_provided_distance and "distance_to_epicenter" in building_df.columns:
    st.info("Using existing distance_to_epicenter column from building_data.")
else:
    if epic_lat is None or epic_lon is None:
        st.warning("No valid epicenter found. You can enter manual coordinates or upload earthquake_data with coordinates.")
        if st.checkbox("Enter manual epicenter coordinates", value=False):
            epic_lat = st.number_input("Manual epicenter latitude", value=0.0, format="%.6f")
            epic_lon = st.number_input("Manual epicenter longitude", value=0.0, format="%.6f")
    if epic_lat is not None and epic_lon is not None:
        building_df["location_lat"] = pd.to_numeric(building_df["location_lat"], errors="coerce")
        building_df["location_lon"] = pd.to_numeric(building_df["location_lon"], errors="coerce")
        missing_coords = building_df["location_lat"].isna() | building_df["location_lon"].isna()
        if missing_coords.any():
            st.warning(f"Dropping {missing_coords.sum()} buildings with missing lat/lon.")
            building_df = building_df.loc[~missing_coords].copy()
        building_df["distance_to_epicenter"] = haversine_vectorized(
            building_df["location_lat"].values,
            building_df["location_lon"].values,
            float(epic_lat),
            float(epic_lon),
        )

# ----------------------
# Merge & preprocessing
# ----------------------
st.header("Data preview & validation")
data = pd.merge(building_df, damage_df, on="building_id", how="inner")
if data.empty:
    st.error("Merge produced empty dataset. Check building_id keys and duplicates.")
    st.stop()

st.subheader("Merged dataset (first rows)")
st.dataframe(data.head(8))

# canonical labels and material code
data["damage_level"] = canonicalize_labels(data["damage_level"])
data["material_type"] = data.get("material_type", pd.Series(["unknown"] * len(data))).fillna("unknown").astype(str)
material_unique = data["material_type"].unique().tolist()
material_map = {m: i for i, m in enumerate(material_unique)}
data["material_code"] = data["material_type"].map(lambda m: material_map.get(str(m), max(material_map.values()) + 1))

st.markdown(f"**Detected material categories:** {material_unique}")

FEATURES = ["age", "distance_to_epicenter", "material_code"]
for f in FEATURES:
    if f not in data.columns:
        st.error(f"Feature {f} missing after preprocessing.")
        st.stop()

for f in FEATURES:
    data[f] = pd.to_numeric(data[f], errors="coerce")

X = data[FEATURES].copy()
y = data["damage_level"].copy()

st.subheader("Class distribution")
st.write(y.value_counts())

# ----------------------
# Model controls
# ----------------------
st.header("Model training & prediction controls")
colA, colB = st.columns([2, 1])
with colB:
    model_options = ["RandomForest", "CalibratedRF", "LogisticRegression"]
    if LGB_AVAILABLE:
        model_options.append("LightGBM")
    if CAT_AVAILABLE:
        model_options.append("CatBoost")
    if NGBOOST_AVAILABLE:
        model_options.append("NGBoostProb")
    selected_model = st.selectbox("Model", model_options)
    use_resample = (
        st.checkbox("Use SMOTETomek resampling (training only)", value=False) if IMBLEARN_AVAILABLE else False
    )
    do_grid = st.checkbox("Run GridSearch (slower)", value=False)
    test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
    random_state = st.number_input("random_state (int)", value=42, step=1)
    n_estimators = st.number_input("n_estimators (trees/iterations)", value=200, step=50)
    model_file = st.text_input("Model filename to save/load", value=f"heritage_model_{selected_model.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    use_class_weight = st.checkbox("Use class_weight='balanced' (fallback if not resampling)", value=True)

with colA:
    st.write("Train / Load")
    train_btn = st.button("Train model")
    load_btn = st.button("Load model from file")

# ----------------------
# Model factory
# ----------------------
def build_model(name, n_estimators, random_state, class_weight=True):
    preproc = ColumnTransformer([("num", StandardScaler(), ["age", "distance_to_epicenter"])], remainder="passthrough")
    if name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state), class_weight="balanced" if class_weight else None)
        return Pipeline([("preproc", preproc), ("clf", clf)])
    if name == "CalibratedRF":
        base = RandomForestClassifier(n_estimators=max(50, int(n_estimators / 2)), random_state=int(random_state), class_weight="balanced" if class_weight else None)
        calib = CalibratedClassifierCV(base_estimator=base, cv=3, method="isotonic")
        return Pipeline([("preproc", preproc), ("clf", calib)])
    if name == "LightGBM" and LGB_AVAILABLE:
        clf = LGBMClassifier(objective="multiclass", n_estimators=int(n_estimators), random_state=int(random_state))
        return Pipeline([("preproc", preproc), ("clf", clf)])
    if name == "CatBoost" and CAT_AVAILABLE:
        clf = CatBoostClassifier(verbose=0, iterations=int(n_estimators), random_state=int(random_state))
        return Pipeline([("preproc", preproc), ("clf", clf)])
    if name == "NGBoostProb" and NGBOOST_AVAILABLE:
        clf = NGBClassifier(n_estimators=int(n_estimators), verbose=False)
        return Pipeline([("preproc", preproc), ("clf", clf)])
    if name == "LogisticRegression":
        clf = LogisticRegression(max_iter=400, class_weight="balanced" if class_weight else None)
        return Pipeline([("preproc", preproc), ("clf", clf)])
    # fallback
    clf = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state), class_weight="balanced" if class_weight else None)
    return Pipeline([("preproc", preproc), ("clf", clf)])


model = None


# ----------------------
# Train / Load handling
# ----------------------
if load_btn:
    if os.path.exists(model_file):
        try:
            loaded = load_artifact(model_file)
            model = loaded.get("model") if isinstance(loaded, dict) else loaded
            mat_map = loaded.get("material_map") if isinstance(loaded, dict) else material_map
            st.success(f"Model loaded from {model_file}")
        except Exception as e:
            st.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
    else:
        st.error(f"Model file {model_file} not found.")

if train_btn:
    try:
        if y.dropna().nunique() < 2:
            st.error("Target variable must contain at least 2 classes for classification.")
            st.stop()
        if len(y.dropna()) < 10:
            st.warning("Very small dataset — results may be unstable.")
        stratify_param = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, stratify=stratify_param, random_state=int(random_state))

        # Resampling (training only)
        if use_resample and IMBLEARN_AVAILABLE:
            try:
                sm = SMOTETomek(random_state=int(random_state))
                X_train, y_train = sm.fit_resample(X_train, y_train)
                st.info(f"Resampled training size: {len(X_train)}")
            except Exception as e:
                st.warning(f"Resampling failed: {e}")

        # Build and train model
        model = build_model(selected_model, n_estimators, random_state, class_weight=use_class_weight)

        if do_grid and selected_model in ["RandomForest", "LightGBM"]:
            param_grid = {"clf__n_estimators": [50, int(n_estimators)]}
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(random_state))
            gs = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            st.info(f"GridSearch best params: {gs.best_params_}")
        else:
            model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, zero_division=0)
        st.success(f"Training complete. Test accuracy: {acc:.3f}, balanced accuracy: {bacc:.3f}")
        st.text("Classification report:")
        st.text(cr)

        # Save artifact
        save_obj = {"model": model, "material_map": material_map, "features": FEATURES}
        try:
            save_artifact(save_obj, model_file)
            st.success(f"Saved model artifact to {model_file}")
        except Exception as e:
            st.error(f"Failed to save model artifact: {e}")
    except Exception as e:
        st.error(f"Training failed: {e}\n{traceback.format_exc()}")
        model = None

# ----------------------
# Predict / show results / map
# ----------------------
st.header("Predictions, diagnostics & map")
# If not trained just now, try to autoload a model file if exists with the chosen name
if model is None and os.path.exists(model_file):
    try:
        loaded = load_artifact(model_file)
        model = loaded.get("model") if isinstance(loaded, dict) else loaded
        st.info(f"Auto-loaded model from {model_file}")
    except Exception:
        model = None

if model is None:
    st.info("No trained model available. Train or load a model to produce predictions and maps.")
    st.stop()

try:
    X_full = data[FEATURES].copy()
    preds = model.predict(X_full)
    data["predicted_damage"] = preds
    if hasattr(model.named_steps["clf"], "predict_proba"):
        try:
            probs = model.predict_proba(X_full)
            data["pred_max_proba"] = probs.max(axis=1)
        except Exception:
            data["pred_max_proba"] = np.nan
    else:
        data["pred_max_proba"] = np.nan

    # Mark uncertainty
    data["pred_uncertain"] = data["pred_max_proba"].fillna(0.0) < 0.7

    st.subheader("Prediction preview")
    display_cols = ["building_id"] + FEATURES + ["predicted_damage", "pred_max_proba", "pred_uncertain"]
    st.dataframe(data[display_cols].head(50))

    # Metrics if labels exist
    if "damage_level" in data.columns:
        st.subheader("Evaluation on merged data")
        st.text(classification_report(data["damage_level"], data["predicted_damage"], zero_division=0))
        cm = confusion_matrix(data["damage_level"], data["predicted_damage"], labels=sorted(data["damage_level"].unique()))
        st.write("Confusion matrix (rows=true, cols=pred):")
        st.write(pd.DataFrame(cm, index=sorted(data["damage_level"].unique()), columns=sorted(data["damage_level"].unique())))

    # Feature importances (if available)
    try:
        clf_obj = model.named_steps["clf"]
        if hasattr(clf_obj, "feature_importances_"):
            fi = clf_obj.feature_importances_
            feat_names = FEATURES
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False)
            st.subheader("Feature importances")
            st.table(fi_df)
    except Exception:
        pass

    # SHAP explanations (optional)
    if SHAP_AVAILABLE:
        try:
            st.subheader("SHAP explanations (approx.)")
            # Try to build an explainer on the underlying estimator (tree or linear)
            est = model.named_steps["clf"]
            preproc = model.named_steps.get("preproc", None)
            X_for_shap = X_full.copy()
            # If pipeline has preprocessor, we pass transformed numeric data into SHAP when needed
            try:
                # If tree model, use TreeExplainer; otherwise use Explainer (model-agnostic)
                if hasattr(est, "feature_importances_"):
                    explainer = shap.TreeExplainer(est)
                else:
                    explainer = shap.Explainer(est.predict, X_for_shap)
                shap_values = explainer(X_for_shap)
                # Summary plot to matplotlib then show
                fig_shap = shap.plots.bar(shap_values, show=False)
                # shap.plots.bar returns a Matplotlib object only in some versions; fallback to summary_plot
                try:
                    plt.tight_layout()
                    st.pyplot(bbox_inches="tight")
                except Exception:
                    # fallback: use shap.summary_plot to produce matplotlib figure
                    plt.figure(figsize=(6, 3))
                    shap.summary_plot(shap_values, X_for_shap, plot_type="bar", show=False)
                    st.pyplot()
            except Exception as e:
                st.warning(f"SHAP plotting failed: {e}")
        except Exception as e:
            st.warning(f"SHAP explanation not available: {e}")

    # Calibration plot if probabilities available and labels exist
    st.subheader("Calibration & probability diagnostics")
    if hasattr(model.named_steps["clf"], "predict_proba") and "damage_level" in data.columns:
        try:
            probs = model.predict_proba(X_full)
            maxp = probs.max(axis=1)
            pred_idx = np.argmax(probs, axis=1)
            cls_order = model.named_steps["clf"].classes_ if hasattr(model.named_steps["clf"], "classes_") else None
            pred_labels = np.array([cls_order[i] for i in pred_idx]) if cls_order is not None else np.array(model.predict(X_full))
            is_correct = (pred_labels == data["damage_level"].values)
            prob_true, prob_pred = calibration_curve(is_correct.astype(int), maxp, n_bins=10)
            fig, ax = plt.subplots()
            ax.plot(prob_pred, prob_true, marker="o", label="max-class correctness")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("Predicted probability (max class)")
            ax.set_ylabel("Observed frequency (correct)")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Calibration plot failed: {e}")
    else:
        st.info("No probability predictions available to plot calibration.")

    # Histogram of max predicted probabilities
    try:
        fig2, ax2 = plt.subplots()
        ax2.hist(data["pred_max_proba"].dropna(), bins=15)
        ax2.set_xlabel("max class probability")
        ax2.set_ylabel("count")
        ax2.set_title("Histogram of max predicted probabilities")
        st.pyplot(fig2)
    except Exception:
        pass

    # Download and map
    st.subheader("Download outputs")
    st.download_button("Download predictions (Excel)", data=to_excel_bytes(data), file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    st.subheader("Interactive map (predicted damage)")
    popup_cols = ["building_id"] + FEATURES + ["predicted_damage", "pred_max_proba"]
    html_map = folium_map_from_df(data, lat_col="location_lat", lon_col="location_lon", label_col="predicted_damage", popup_cols=popup_cols)
    components.html(html_map, height=650, scrolling=True)

except Exception as e:
    st.error(f"Error when predicting or rendering results: {e}\n{traceback.format_exc()}")

# Footer
st.markdown("---")
st.markdown("<small>App: enhanced single-file deployment demo. Validate models and decisions before production use.</small>", unsafe_allow_html=True)
