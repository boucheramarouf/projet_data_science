"""
Streamlit dashboard — Injury Prediction for Competitive Runners.

Pages:
  1. Overview & KPIs
  2. Model Comparison
  3. Real-Time Prediction
  4. SHAP Interpretability
  5. Simulation (What-If)
"""

import os, sys, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    average_precision_score, roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.preprocessing import load_raw_data, preprocess, split_data, add_aggregate_features
from src.models import load_sklearn_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Injury Prediction — Runners",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "svm":                 "SVM",
    "mlp":                 "MLP (Deep Learning)",
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des données...")
def load_data():
    df = load_raw_data()
    X, y, feat_names = preprocess(df, add_features=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, feat_names, df


@st.cache_resource(show_spinner="Chargement des modèles...")
def load_all_models():
    out = {}
    for name in DISPLAY_NAMES:
        path = os.path.join(ROOT, "saved_models", f"{name}.pkl")
        if os.path.exists(path):
            out[name] = load_sklearn_model(name)
    return out


@st.cache_data
def load_thresholds():
    p = os.path.join(ROOT, "results", "thresholds.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data
def load_comparison():
    p = os.path.join(ROOT, "results", "model_comparison.csv")
    return pd.read_csv(p, index_col="model") if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏃 Injury Prediction")
    st.caption("RNCP40875 — Bloc 2")
    page = st.radio("Navigation", [
        "Overview", "Model Comparison",
        "Real-Time Prediction", "SHAP Interpretability", "Simulation"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("42 766 obs · 73 colonnes · 7-day windows")
    st.markdown("Taux de blessure ≈ 1.4 %")

# ---------------------------------------------------------------------------
# Guard — models must exist
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(ROOT, "saved_models", "random_forest.pkl")):
    st.error("Modèles non trouvés. Lancez d'abord : `python train.py`")
    st.stop()

X_train, X_test, y_train, y_test, feat_names, df_raw = load_data()
models   = load_all_models()
thresh   = load_thresholds()   # {"Logistic Regression": 0.61, ...}
comp_csv = load_comparison()


def get_proba(key):
    return models[key].predict_proba(X_test)[:, 1]


def get_threshold(key):
    display = DISPLAY_NAMES[key]
    # Try full display name first, then short name (handles "MLP" vs "MLP (Deep Learning)")
    return thresh.get(display, thresh.get(display.split(" (")[0], 0.5))


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Tableau de bord — Prédiction de blessures")
    st.markdown(
        "Prédiction binaire : un coureur compétitif va-t-il se blesser **la semaine suivante** "
        "en fonction de sa charge d'entraînement sur **7 jours** ?"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df_raw):,}")
    c2.metric("Features", len(feat_names))
    c3.metric("Taux de blessure", f"{df_raw['injury'].mean():.2%}")
    c4.metric("Semaines blessées", f"{int(df_raw['injury'].sum()):,}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution des classes")
        counts = df_raw["injury"].value_counts().reset_index()
        counts.columns = ["Label", "Count"]
        counts["Label"] = counts["Label"].map({0: "Non blessé", 1: "Blessé"})
        fig = px.pie(counts, names="Label", values="Count",
                     color="Label",
                     color_discrete_map={"Non blessé": "#4CAF50", "Blessé": "#F44336"},
                     hole=0.4)
        fig.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Km totaux (jour 0) par statut")
        col_km = "total km"
        fig2 = px.histogram(
            df_raw, x=col_km,
            color=df_raw["injury"].map({0: "Non blessé", 1: "Blessé"}),
            nbins=60, barmode="overlay", opacity=0.7,
            color_discrete_map={"Non blessé": "#4CAF50", "Blessé": "#F44336"},
            labels={"color": "Statut", col_km: "Total km (jour 0)"}
        )
        fig2.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Corrélations — features agrégées vs blessure")
    agg_cols = [c for c in feat_names if c.startswith("agg_sum")][:10]
    if agg_cols:
        corr_df = X_train[agg_cols].copy()
        corr_df["injury"] = y_train.values
        corr = corr_df.corr()
        fig3 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=".2f", aspect="auto")
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Model Comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Comparaison des modèles")
    st.markdown("5 modèles évalués sur le jeu de test (20 %) avec seuil de décision optimisé.")

    if comp_csv is not None:
        cols_show = ["f1_injury", "recall_injury", "precision_injury", "pr_auc", "roc_auc", "threshold"]
        st.dataframe(
            comp_csv[cols_show].style
            .highlight_max(subset=["f1_injury", "recall_injury", "pr_auc", "roc_auc"], color="#d4edda")
            .format("{:.4f}"),
            use_container_width=True,
        )

    st.divider()

    # Courbes PR
    st.subheader("Courbes Précision-Rappel")
    fig_pr = go.Figure()
    fig_pr.add_hline(y=float(y_test.mean()), line_dash="dash", line_color="gray",
                     annotation_text=f"Baseline ({y_test.mean():.3f})")
    for key in models:
        yp = get_proba(key)
        prec, rec, _ = precision_recall_curve(y_test, yp)
        auc = average_precision_score(y_test, yp)
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                    name=f"{DISPLAY_NAMES[key]} (AUC={auc:.3f})"))
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                         xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig_pr, use_container_width=True)

    # Courbes ROC
    st.subheader("Courbes ROC")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"), name="Aléatoire"))
    for key in models:
        yp = get_proba(key)
        fpr, tpr, _ = roc_curve(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"{DISPLAY_NAMES[key]} (AUC={auc:.3f})"))
    fig_roc.update_layout(xaxis_title="Taux faux positifs", yaxis_title="Taux vrais positifs",
                           xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig_roc, use_container_width=True)

    # Matrices de confusion
    st.subheader("Matrices de confusion")
    cols_cm = st.columns(len(models))
    for i, key in enumerate(models):
        with cols_cm[i]:
            yp = get_proba(key)
            t = get_threshold(key)
            ypred = (yp >= t).astype(int)
            cm = confusion_matrix(y_test, ypred)
            fig_cm = px.imshow(cm, labels=dict(x="Prédit", y="Réel"),
                               x=["Non blessé", "Blessé"], y=["Non blessé", "Blessé"],
                               text_auto=True, color_continuous_scale="Blues",
                               title=DISPLAY_NAMES[key])
            fig_cm.update_layout(height=260, margin=dict(t=40, b=0))
            st.plotly_chart(fig_cm, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3 — Real-Time Prediction
# ---------------------------------------------------------------------------
elif page == "Real-Time Prediction":
    st.title("Prédiction du risque en temps réel")
    st.markdown("Saisissez la charge d'entraînement d'un coureur sur **7 jours** pour obtenir son score de risque.")

    DAYS   = ["Jour -6 (ancien)", "Jour -5", "Jour -4", "Jour -3", "Jour -2", "Jour -1", "Jour 0 (aujourd'hui)"]
    METRICS = ["nr. sessions", "total km", "km Z3-4", "km Z5-T1-T2",
               "km sprinting", "strength training", "hours alternative",
               "perceived exertion", "perceived trainingSuccess", "perceived recovery"]

    sel_model = st.selectbox("Modèle", list(models.keys()), format_func=lambda k: DISPLAY_NAMES[k])

    with st.expander("Données d'entraînement (7 jours)", expanded=True):
        input_data = {}
        tabs = st.tabs(DAYS)
        for d_idx, tab in enumerate(tabs):
            with tab:
                c1, c2 = st.columns(2)
                for m_idx, metric in enumerate(METRICS):
                    col = c1 if m_idx % 2 == 0 else c2
                    suffix = f".{d_idx}" if d_idx > 0 else ""
                    key_id = f"{metric}{suffix}_inp"
                    if metric == "strength training":
                        val = col.selectbox(metric, [0, 1], key=key_id)
                    elif "perceived" in metric:
                        val = col.slider(metric, 0.0, 1.0, 0.1, step=0.01, key=key_id)
                    elif metric == "nr. sessions":
                        val = col.number_input(metric, 0, 10, 1, key=key_id)
                    else:
                        val = col.number_input(metric, 0.0, 150.0, 0.0, step=0.5, key=key_id)
                    col_name = f"{metric}{suffix}"
                    input_data[col_name] = float(val)

    if st.button("Prédire le risque", type="primary", use_container_width=True):
        # Build row aligned to feat_names
        row = {f: input_data.get(f, 0.0) for f in feat_names}
        row_df = pd.DataFrame([row])
        row_full = add_aggregate_features(row_df)
        for c in feat_names:
            if c not in row_full.columns:
                row_full[c] = 0.0
        row_final = row_full[list(feat_names)]

        prob = float(models[sel_model].predict_proba(row_final)[0, 1])
        t = get_threshold(sel_model)
        is_high = prob >= t

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité de blessure", f"{prob:.1%}")
        c2.metric("Seuil de décision", f"{t:.2f}")
        c3.metric("Niveau de risque", "RISQUE ELEVE" if is_high else "RISQUE FAIBLE")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#F44336" if is_high else "#4CAF50"},
                "steps": [
                    {"range": [0, t * 100], "color": "#e8f5e9"},
                    {"range": [t * 100, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": t * 100},
            },
            title={"text": "Score de risque de blessure"},
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=0))
        st.plotly_chart(gauge, use_container_width=True)

        if is_high:
            st.error("**Alerte :** Ce coureur présente un risque élevé de blessure la semaine prochaine. "
                     "Réduisez le volume ou l'intensité d'entraînement.")
        else:
            st.success("Risque dans les limites acceptables. Continuez à surveiller la charge hebdomadaire.")


# ---------------------------------------------------------------------------
# Page 4 — SHAP Interpretability
# ---------------------------------------------------------------------------
elif page == "SHAP Interpretability":
    st.title("Interprétabilité SHAP")
    st.markdown(
        "Les valeurs SHAP expliquent **pourquoi** le modèle prédit un risque élevé ou faible, "
        "et quelles features y contribuent le plus."
    )

    # Prefer RF, fallback to XGBoost
    for model_label, bar_file, bee_file, csv_file in [
        ("Random Forest",
         "shap_summary_Random_Forest.png", "shap_beeswarm_Random_Forest.png", "shap_top_features_rf.csv"),
        ("XGBoost",
         "shap_summary_XGBoost.png", "shap_beeswarm_XGBoost.png", "shap_top_features_xgb.csv"),
    ]:
        bar_path = os.path.join(ROOT, "figures", bar_file)
        bee_path = os.path.join(ROOT, "figures", bee_file)
        csv_path = os.path.join(ROOT, "results", csv_file)
        if os.path.exists(bar_path):
            st.subheader(f"Modèle : {model_label}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Importance globale (mean |SHAP|)**")
                st.image(bar_path)
            with c2:
                st.markdown("**Beeswarm — impact individuel**")
                st.image(bee_path)

            if os.path.exists(csv_path):
                df_shap = pd.read_csv(csv_path)
                st.subheader("Top 20 features les plus influentes")
                fig_shap = px.bar(
                    df_shap.head(20),
                    x="mean_abs_shap", y="feature", orientation="h",
                    labels={"mean_abs_shap": "Valeur SHAP moyenne |absolue|", "feature": "Feature"},
                    color="mean_abs_shap", color_continuous_scale="Oranges"
                )
                fig_shap.update_layout(yaxis={"categoryorder": "total ascending"}, height=550)
                st.plotly_chart(fig_shap, use_container_width=True)
            st.divider()

    st.subheader("Comment lire les valeurs SHAP ?")
    st.markdown("""
| Valeur SHAP | Signification |
|---|---|
| **Positive** | La feature augmente le risque de blessure |
| **Négative** | La feature réduit le risque de blessure |
| **Grande valeur absolue** | Feature très influente sur la prédiction |

- **Bar chart** : importance globale moyenne (sur tous les coureurs du jeu de test)
- **Beeswarm** : chaque point est une observation — rouge = valeur élevée, bleu = valeur faible
""")


# ---------------------------------------------------------------------------
# Page 5 — Simulation
# ---------------------------------------------------------------------------
elif page == "Simulation":
    st.title("Simulation — Analyse What-If")
    st.markdown(
        "Simulez l'impact d'une **réduction de charge d'entraînement** sur le risque de blessure prédit."
    )

    sel_model_sim = st.selectbox("Modèle", list(models.keys()),
                                  format_func=lambda k: DISPLAY_NAMES[k], key="sim_model")

    st.subheader("Profil de charge hebdomadaire de base")
    c1, c2, c3 = st.columns(3)
    base_km       = c1.slider("Km totaux / semaine", 0.0, 150.0, 60.0, step=5.0)
    base_sessions = c2.slider("Sessions / semaine", 1, 14, 6)
    base_intensity= c3.slider("Km haute intensité (Z5-T1-T2)", 0.0, 30.0, 5.0, step=0.5)

    reduction_pct = st.slider("Réduction de volume/intensité (%)", 0, 60, 0, step=5)

    def build_row(km_factor, int_factor):
        row = {}
        for feat in feat_names:
            if feat.startswith("agg_") or feat == "acwr_total_km":
                row[feat] = 0.0
            elif "total km" in feat:
                row[feat] = base_km / 7 * km_factor
            elif "nr. sessions" in feat:
                row[feat] = base_sessions / 7
            elif "km Z5-T1-T2" in feat:
                row[feat] = base_intensity / 7 * int_factor
            elif "km Z3-4" in feat:
                row[feat] = base_intensity / 7 * 0.5 * int_factor
            elif "km sprinting" in feat:
                row[feat] = base_km / 7 * 0.1 * km_factor
            elif "perceived" in feat:
                row[feat] = max(0.05, 0.3 * km_factor + 0.1)
            else:
                row[feat] = 0.0
        row_df = pd.DataFrame([row])
        row_full = add_aggregate_features(row_df)
        for c in feat_names:
            if c not in row_full.columns:
                row_full[c] = 0.0
        return row_full[list(feat_names)]

    if st.button("Lancer la simulation", type="primary"):
        factor = 1.0 - reduction_pct / 100.0
        row_final = build_row(factor, factor)
        prob = float(models[sel_model_sim].predict_proba(row_final)[0, 1])
        t = get_threshold(sel_model_sim)

        c1, c2 = st.columns(2)
        c1.metric(f"Probabilité avec -{reduction_pct}% de charge", f"{prob:.1%}")
        c2.metric("Seuil d'alerte", f"{t:.0%}")

        # Courbe de réduction
        reductions = list(range(0, 65, 5))
        probs = []
        for red in reductions:
            f = 1.0 - red / 100.0
            p = float(models[sel_model_sim].predict_proba(build_row(f, f))[0, 1])
            probs.append(p)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=reductions, y=[p * 100 for p in probs],
            mode="lines+markers", name="Risque de blessure",
            line=dict(color="#F44336", width=2),
        ))
        fig_sim.add_hline(y=t * 100, line_dash="dash", line_color="orange",
                           annotation_text=f"Seuil ({t:.0%})")
        fig_sim.update_layout(
            xaxis_title="Réduction de charge (%)",
            yaxis_title="Probabilité de blessure (%)",
            title="Impact d'une réduction de charge sur le risque de blessure",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        safe = next((r for r, p in zip(reductions, probs) if p < t), None)
        if safe is not None:
            st.success(f"Une réduction de **{safe}%** ramène le risque sous le seuil d'alerte ({t:.0%}).")
        else:
            st.warning("La réduction de charge seule ne suffit pas. Envisagez aussi de réduire les séances haute intensité.")
