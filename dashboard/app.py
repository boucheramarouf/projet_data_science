"""
RunSafe AI — Injury Prediction Dashboard
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

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RunSafe AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation state ──────────────────────────────────────────────────────────
NAV_ITEMS = [
    "Vue d'ensemble",
    "Comparaison des modèles",
    "Prédiction",
    "Interprétabilité SHAP",
    "Simulation",
]
if "page" not in st.session_state:
    st.session_state.page = NAV_ITEMS[0]

active_idx = NAV_ITEMS.index(st.session_state.page)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }}
.stApp {{ background: #f8fafc !important; }}

/* ── Strip Streamlit chrome ───────────────────────────── */
header[data-testid="stHeader"] {{ display: none !important; }}
.block-container {{ padding-top: 0 !important; padding-bottom: 32px !important; max-width: 100% !important; }}

/* ── Sidebar base ─────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: #0f172a !important;
    padding-top: 0 !important;
}}

/* ── Nav buttons — all inactive ───────────────────────── */
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: .01em !important;
    padding: 10px 16px !important;
    text-align: left !important;
    text-transform: none !important;
    width: 100% !important;
    justify-content: flex-start !important;
    cursor: pointer !important;
    transition: color .12s, background .12s !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,.05) !important;
    color: #cbd5e1 !important;
    border-left: 3px solid #334155 !important;
    box-shadow: none !important;
}}

/* ── Active nav button — nth-of-type trick ────────────── */
[data-testid="stSidebar"] .stButton:nth-of-type({active_idx + 1}) > button {{
    background: rgba(37,99,235,.14) !important;
    color: #93c5fd !important;
    border-left: 3px solid #2563eb !important;
    font-weight: 600 !important;
}}

/* ── Primary button (outside sidebar) ────────────────── */
.main .stButton > button {{
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    transition: background .15s !important;
    width: 100% !important;
    box-shadow: none !important;
}}
.main .stButton > button:hover {{ background: #1d4ed8 !important; box-shadow: none !important; }}

/* ── Typography ───────────────────────────────────────── */
h1 {{ font-size: 20px !important; font-weight: 700 !important;
     color: #0f172a !important; letter-spacing: -.3px !important; margin: 0 !important; }}
h2 {{ font-size: 15px !important; font-weight: 600 !important; color: #1e293b !important; }}
h3 {{ font-size: 13px !important; font-weight: 600 !important; color: #475569 !important; }}

/* ── Tabs ─────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
    background: #f1f5f9 !important; border-radius: 5px !important;
    padding: 3px !important; gap: 2px !important; border: none !important;
}}
[data-baseweb="tab"] {{
    border-radius: 4px !important; font-size: 12px !important;
    font-weight: 500 !important; color: #64748b !important;
    padding: 5px 12px !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: #fff !important; color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.07) !important;
}}

/* ── Metrics ──────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: #fff !important; border-radius: 6px !important;
    padding: 16px 18px !important; border: 1px solid #e2e8f0 !important;
}}
[data-testid="stMetricValue"] {{
    color: #0f172a !important; font-weight: 700 !important; font-size: 24px !important;
}}
[data-testid="stMetricLabel"] {{ color: #64748b !important; font-size: 11px !important; }}

/* ── Dataframe ────────────────────────────────────────── */
[data-testid="stDataFrame"] {{ border-radius: 6px !important; overflow: hidden !important; }}

/* ── Inputs ───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {{ border-radius: 5px !important; }}
.stSlider > div > div > div {{ color: #2563eb !important; }}
.stNumberInput input {{ border-radius: 5px !important; font-size: 13px !important; }}

/* Divider */
hr {{ border-color: #e2e8f0 !important; margin: 20px 0 !important; }}
[data-testid="stAlert"] {{ border-radius: 6px !important; }}
.stCaption {{ color: #94a3b8 !important; font-size: 11px !important; }}

/* ── Expander ─────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    background: #fff !important;
}}

/* ── Selectbox ────────────────────────────────────────── */
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #94a3b8 !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Chart palette & helpers ────────────────────────────────────────────────────
C = ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed", "#db2777"]

_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#374151"),
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    colorway=C,
    margin=dict(l=44, r=16, t=44, b=36),
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False, tickfont=dict(size=11)),
    title_font=dict(size=14, color="#1e293b", family="Inter, system-ui"),
    hoverlabel=dict(bgcolor="#fff", bordercolor="#e2e8f0", font_size=12,
                    font_family="Inter, system-ui"),
    legend=dict(bgcolor="rgba(255,255,255,.95)", bordercolor="#e2e8f0",
                borderwidth=1, font=dict(size=11)),
)

def _fig(fig, title="", h=None):
    fig.update_layout(**_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color="#1e293b")))
    if h:
        fig.update_layout(height=h)
    return fig


# ── HTML primitives ────────────────────────────────────────────────────────────

def kpi(label, value, note="", accent="#2563eb"):
    return f"""
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;
     border-left:4px solid {accent};padding:18px 20px;height:100%">
  <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
       letter-spacing:.08em;margin-bottom:10px">{label}</div>
  <div style="font-size:28px;font-weight:700;color:#0f172a;line-height:1">{value}</div>
  <div style="font-size:11px;color:#94a3b8;margin-top:8px">{note}</div>
</div>"""

def page_title(title, sub=""):
    s = (f'<p style="font-size:13px;color:#64748b;margin:4px 0 0;font-weight:400">{sub}</p>'
         if sub else "")
    return f"""
<div style="padding:28px 0 22px;border-bottom:1px solid #e2e8f0;margin-bottom:28px">
  <h1>{title}</h1>{s}
</div>"""

def section(title=""):
    t = (f'<div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
         f'letter-spacing:.08em;margin-bottom:14px;padding-bottom:10px;'
         f'border-bottom:1px solid #f1f5f9">{title}</div>' if title else "")
    return f'<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;padding:20px">{t}'

def section_end():
    return '</div>'

def panel(content="", title="", pa="20px"):
    t = (f'<div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
         f'letter-spacing:.08em;margin-bottom:14px;padding-bottom:10px;'
         f'border-bottom:1px solid #f1f5f9">{title}</div>' if title else "")
    return (f'<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;'
            f'padding:{pa}">{t}{content}</div>')

def risk_banner(is_high, prob, thr):
    if is_high:
        return f"""
<div style="background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;
     border-radius:8px;padding:18px 22px">
  <div style="font-size:10px;font-weight:700;color:#dc2626;text-transform:uppercase;
       letter-spacing:.08em;margin-bottom:8px">Risque elevé — intervention recommandée</div>
  <div style="font-size:15px;font-weight:600;color:#991b1b">
    Probabilité : {prob:.1%}&nbsp;&nbsp;·&nbsp;&nbsp;Seuil : {thr:.0%}
  </div>
  <div style="margin-top:12px;padding-top:12px;border-top:1px solid #fecaca;
       font-size:13px;color:#7f1d1d;line-height:1.7">
    Réduire le volume d'entraînement de 20–30 %, limiter les séances haute
    intensité et augmenter la récupération active avant la prochaine semaine.
  </div>
</div>"""
    return f"""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #059669;
     border-radius:8px;padding:18px 22px">
  <div style="font-size:10px;font-weight:700;color:#059669;text-transform:uppercase;
       letter-spacing:.08em;margin-bottom:8px">Risque faible — charge acceptable</div>
  <div style="font-size:15px;font-weight:600;color:#166534">
    Probabilité : {prob:.1%}&nbsp;&nbsp;·&nbsp;&nbsp;Seuil : {thr:.0%}
  </div>
  <div style="margin-top:12px;padding-top:12px;border-top:1px solid #bbf7d0;
       font-size:13px;color:#14532d;line-height:1.7">
    Charge d'entraînement dans les limites acceptables.
    Surveiller la récupération perçue et le ratio charge aiguë/chronique.
  </div>
</div>"""

def stat_row(items):
    rows = "".join(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:8px 0;border-bottom:1px solid #1e293b">
  <span style="font-size:12px;color:#64748b">{k}</span>
  <span style="font-size:12px;font-weight:600;color:#94a3b8">{v}</span>
</div>""" for k, v in items)
    return f'<div>{rows}</div>'

def empty_state(msg):
    return f"""
<div style="padding:64px 20px;text-align:center;background:#fff;
     border-radius:8px;border:1px solid #e2e8f0">
  <div style="width:44px;height:44px;border-radius:50%;background:#f1f5f9;
       margin:0 auto 16px;line-height:44px;text-align:center">
    <div style="width:20px;height:2px;background:#cbd5e1;border-radius:1px;
         display:inline-block;vertical-align:middle"></div>
  </div>
  <div style="font-size:14px;font-weight:500;color:#64748b">{msg}</div>
</div>"""


# ── Model registry ─────────────────────────────────────────────────────────────
KEYS = ["logistic_regression", "random_forest", "xgboost", "svm", "mlp"]
LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "svm":                 "SVM",
    "mlp":                 "MLP",
}


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_data():
    df = load_raw_data()
    X, y, feats = preprocess(df, add_features=True)
    return *split_data(X, y), feats, df

@st.cache_resource(show_spinner=False)
def _load_models():
    return {k: load_sklearn_model(k)
            for k in KEYS
            if os.path.exists(os.path.join(ROOT, "saved_models", f"{k}.pkl"))}

@st.cache_data
def _load_thresholds():
    p = os.path.join(ROOT, "results", "thresholds.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_data
def _load_comparison():
    p = os.path.join(ROOT, "results", "model_comparison.csv")
    return pd.read_csv(p, index_col="model") if os.path.exists(p) else None

@st.cache_data
def _load_cv():
    p = os.path.join(ROOT, "results", "cv_results.csv")
    return pd.read_csv(p, index_col="model") if os.path.exists(p) else None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
<div style="padding:22px 16px 16px">
  <div style="font-size:16px;font-weight:700;color:#f1f5f9;letter-spacing:-.2px">
    RunSafe AI
  </div>
  <div style="font-size:10px;color:#334155;margin-top:3px;text-transform:uppercase;
       letter-spacing:.07em">Injury Prediction System</div>
</div>
<div style="height:1px;background:#1e293b;margin:0 0 8px"></div>
""", unsafe_allow_html=True)

    # Navigation buttons — session_state driven
    for item in NAV_ITEMS:
        if st.button(item, key=f"_nav_{item}", use_container_width=True):
            st.session_state.page = item
            st.rerun()

    # Dataset stats
    st.markdown("""
<div style="height:1px;background:#1e293b;margin:16px 0 12px"></div>
<div style="font-size:10px;font-weight:700;color:#334155;text-transform:uppercase;
     letter-spacing:.07em;padding:0 16px;margin-bottom:8px">Dataset</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style="padding:0 16px">
""" + stat_row([
        ("Observations", "42 766"),
        ("Features", "73"),
        ("Train / Test", "80 % / 20 %"),
        ("Taux de blessure", "1.4 %"),
        ("Modèles", "5"),
    ]) + """</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin:20px 16px 0;font-size:11px;color:#334155;line-height:1.8">
  RNCP40875 &middot; Bloc 2 &middot; 2025-26<br>
  B. Marouf &amp; M. Touati — EFREI Paris
</div>""", unsafe_allow_html=True)


# ── Guard ──────────────────────────────────────────────────────────────────────
if not os.path.exists(os.path.join(ROOT, "saved_models", "random_forest.pkl")):
    st.error("Modèles introuvables. Exécutez : python train.py")
    st.stop()

with st.spinner("Chargement des données…"):
    X_train, X_test, y_train, y_test, feat_names, df_raw = _load_data()
    models  = _load_models()
    thresh  = _load_thresholds()
    comp    = _load_comparison()
    cv_data = _load_cv()

page = st.session_state.page

def _prob(k):
    return models[k].predict_proba(X_test)[:, 1]

def _thr(k):
    d = LABELS[k]
    return thresh.get(d, thresh.get(d.split("(")[0].strip(), 0.5))


# ══════════════════════════════════════════════════════════════════════════════
#  VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Vue d'ensemble":
    st.markdown(page_title(
        "Vue d'ensemble",
        "Dataset · Fenêtres glissantes 7 jours · Coureurs compétitifs"
    ), unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.markdown(kpi("Observations", "42 766", "Fenêtres glissantes 7 jours", "#2563eb"),
                unsafe_allow_html=True)
    c2.markdown(kpi("Features", str(len(feat_names)), "Après feature engineering", "#0891b2"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Taux de blessure", f"{df_raw['injury'].mean():.2%}",
                    "Déséquilibre sévère — PR-AUC prioritaire", "#dc2626"),
                unsafe_allow_html=True)
    c4.markdown(kpi("Meilleur PR-AUC", "0.032",
                    "Random Forest · x2.3 vs baseline (0.014)", "#059669"),
                unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Charts row
    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.markdown(panel(title="Répartition des classes"), unsafe_allow_html=True)
        counts = df_raw["injury"].value_counts().reset_index()
        counts.columns = ["Label", "Count"]
        counts["Label"] = counts["Label"].map({0: "Non blessé", 1: "Blessé"})
        fig = px.pie(counts, names="Label", values="Count", hole=0.60,
                     color="Label",
                     color_discrete_map={"Non blessé": "#2563eb", "Blessé": "#dc2626"})
        fig.update_traces(textinfo="percent+label", textfont_size=11,
                          textposition="outside", pull=[0, 0.05])
        _fig(fig)
        fig.update_layout(
            showlegend=False, height=270,
            margin=dict(l=8, r=8, t=20, b=8),
            annotations=[dict(text="Classes", x=0.5, y=0.5,
                              font=dict(size=12, color="#94a3b8"), showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(panel(title="Distribution des km totaux par statut"), unsafe_allow_html=True)
        fig2 = px.histogram(
            df_raw, x="total km",
            color=df_raw["injury"].map({0: "Non blessé", 1: "Blessé"}),
            nbins=50, barmode="overlay", opacity=0.75,
            color_discrete_map={"Non blessé": "#2563eb", "Blessé": "#dc2626"},
            labels={"color": "Statut", "total km": "Km totaux (jour J)"},
        )
        _fig(fig2)
        fig2.update_layout(height=270, margin=dict(l=40, r=10, t=20, b=36),
                            legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(panel(title="Corrélations — features agrégées 7 jours vs blessure"),
                unsafe_allow_html=True)
    agg_cols = [c for c in feat_names if c.startswith("agg_sum")][:10]
    if agg_cols:
        corr_df = X_train[agg_cols].copy()
        corr_df["injury"] = y_train.values
        fig3 = px.imshow(corr_df.corr(), color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        _fig(fig3)
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                            coloraxis_colorbar=dict(thickness=10, len=0.7,
                                                     tickfont=dict(size=10)))
        st.plotly_chart(fig3, use_container_width=True)

    # Context card
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;
     border-left:4px solid #2563eb;padding:18px 22px">
  <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
       letter-spacing:.08em;margin-bottom:10px">Contexte métier</div>
  <p style="font-size:13px;color:#475569;margin:0;line-height:1.8">
    Ce système prédit le risque de blessure d'un coureur compétitif pour la semaine suivante,
    à partir de sa charge d'entraînement sur les <strong>7 derniers jours</strong>.
    Le fort déséquilibre des classes (1.4 % de blessures) impose d'utiliser la
    <strong>PR-AUC</strong> et le <strong>Recall</strong> comme métriques prioritaires —
    l'accuracy étant trompeuse dans ce contexte.
  </p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARAISON DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Comparaison des modèles":
    st.markdown(page_title(
        "Comparaison des modèles",
        "5 algorithmes · jeu de test 20 % · seuil de décision optimisé par F1"
    ), unsafe_allow_html=True)

    if comp is not None:
        st.markdown(panel(title="Tableau des performances"), unsafe_allow_html=True)
        rename = {
            "accuracy": "Accuracy", "f1_injury": "F1 (blessé)",
            "recall_injury": "Rappel", "precision_injury": "Précision",
            "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "threshold": "Seuil",
        }
        show = ["accuracy", "f1_injury", "recall_injury", "precision_injury",
                "pr_auc", "roc_auc", "threshold"]
        styled = (comp[show]
                  .rename(columns=rename)
                  .style
                  .highlight_max(subset=["F1 (blessé)", "Rappel", "PR-AUC", "ROC-AUC"],
                                  color="#dbeafe")
                  .highlight_min(subset=["F1 (blessé)", "PR-AUC"], color="#fee2e2")
                  .format("{:.4f}"))
        st.dataframe(styled, use_container_width=True)
        st.caption("Bleu = meilleure valeur   ·   Rouge = valeur la plus faible")

    # Radar chart
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if comp is not None:
        st.markdown(panel(title="Radar — performance multi-dimensionnelle"), unsafe_allow_html=True)
        mets = ["f1_injury", "recall_injury", "precision_injury", "pr_auc", "roc_auc"]
        labs = ["F1 (blessé)", "Rappel", "Précision", "PR-AUC", "ROC-AUC"]
        fig_r = go.Figure()
        for i, (mn, row) in enumerate(comp.iterrows()):
            vals = [row[m] for m in mets] + [row[mets[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=labs + [labs[0]], name=mn, fill="toself", opacity=0.35,
                line=dict(color=C[i % len(C)], width=2),
            ))
        _fig(fig_r)
        fig_r.update_layout(
            height=380, margin=dict(l=60, r=60, t=20, b=20),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#e2e8f0", tickfont=dict(size=10)),
                angularaxis=dict(gridcolor="#e2e8f0"),
                bgcolor="#fff",
            ),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Cross-validation table
    if cv_data is not None:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(panel(title="Validation croisée — 5-fold stratifiée (μ ± σ)"),
                    unsafe_allow_html=True)
        rcv = {
            "f1_mean": "F1 μ", "f1_std": "F1 σ",
            "recall_mean": "Rappel μ", "recall_std": "Rappel σ",
            "pr_auc_mean": "PR-AUC μ", "pr_auc_std": "PR-AUC σ",
            "roc_auc_mean": "ROC-AUC μ", "roc_auc_std": "ROC-AUC σ",
        }
        st.dataframe(cv_data.rename(columns=rcv).style.format("{:.4f}"),
                     use_container_width=True)
        st.caption("Faibles écarts-types → résultats stables, non spécifiques au split unique.")
    else:
        st.info("Résultats de validation croisée non disponibles. Exécutez : python run_cv.py")

    # PR + ROC curves
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_pr, col_roc = st.columns(2, gap="medium")
    with col_pr:
        st.markdown(panel(title="Courbes précision-rappel"), unsafe_allow_html=True)
        fig_pr = go.Figure()
        bl = float(y_test.mean())
        fig_pr.add_hline(y=bl, line_dash="dot", line_color="#cbd5e1",
                          annotation_text=f"Baseline ({bl:.3f})",
                          annotation_font=dict(size=10, color="#94a3b8"))
        for i, k in enumerate(models):
            yp = _prob(k)
            prec, rec, _ = precision_recall_curve(y_test, yp)
            auc = average_precision_score(y_test, yp)
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=f"{LABELS[k]} ({auc:.3f})",
                line=dict(color=C[i], width=2),
            ))
        _fig(fig_pr)
        fig_pr.update_layout(xaxis_title="Rappel", yaxis_title="Précision",
                               xaxis_range=[0, 1], yaxis_range=[0, 1], height=340)
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_roc:
        st.markdown(panel(title="Courbes ROC"), unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dot", color="#e2e8f0", width=1),
            showlegend=False,
        ))
        for i, k in enumerate(models):
            yp = _prob(k)
            fpr, tpr, _ = roc_curve(y_test, yp)
            auc = roc_auc_score(y_test, yp)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{LABELS[k]} ({auc:.3f})",
                line=dict(color=C[i], width=2),
            ))
        _fig(fig_roc)
        fig_roc.update_layout(xaxis_title="Taux faux positifs",
                               yaxis_title="Taux vrais positifs",
                               xaxis_range=[0, 1], yaxis_range=[0, 1], height=340)
        st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrices
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(panel(title="Matrices de confusion — seuil optimisé F1"), unsafe_allow_html=True)
    cols_cm = st.columns(len(models), gap="small")
    for i, k in enumerate(models):
        with cols_cm[i]:
            yp = _prob(k)
            t = _thr(k)
            ypred = (yp >= t).astype(int)
            cm = confusion_matrix(y_test, ypred)
            fig_cm = px.imshow(
                cm, text_auto=True,
                color_continuous_scale=[[0, "#f8fafc"], [1, C[i]]],
                labels=dict(x="Prédit", y="Réel"),
                x=["Négatif", "Positif"], y=["Négatif", "Positif"],
            )
            fig_cm.update_layout(
                title=dict(text=LABELS[k], font=dict(size=11, color="#1e293b")),
                height=220, margin=dict(l=8, r=8, t=32, b=8),
                paper_bgcolor="#fff", font=dict(family="Inter", size=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prédiction":
    st.markdown(page_title(
        "Prédiction du risque",
        "Saisissez la charge d'entraînement hebdomadaire pour évaluer le risque de blessure"
    ), unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    DAYS_LBL = ["J-6", "J-5", "J-4", "J-3", "J-2", "J-1", "J0 (aujourd'hui)"]
    METRICS = [
        "nr. sessions", "total km", "km Z3-4", "km Z5-T1-T2",
        "km sprinting", "strength training", "hours alternative",
        "perceived exertion", "perceived trainingSuccess", "perceived recovery",
    ]

    with col_form:
        st.markdown(panel(title="Données d'entraînement — 7 jours"), unsafe_allow_html=True)

        sel_model = st.selectbox(
            "Modèle de prédiction",
            list(models.keys()),
            format_func=lambda k: LABELS[k] + (" (recommandé)" if k == "random_forest" else ""),
        )

        input_data: dict = {}
        tabs = st.tabs(DAYS_LBL)
        for d_idx, tab in enumerate(tabs):
            with tab:
                c1, c2 = st.columns(2)
                for m_idx, metric in enumerate(METRICS):
                    col = c1 if m_idx % 2 == 0 else c2
                    kid = f"inp_{metric}_{d_idx}"
                    if metric == "strength training":
                        val = col.selectbox(metric, [0, 1], key=kid)
                    elif "perceived" in metric:
                        val = col.slider(metric, 0.0, 1.0, 0.1, 0.01, key=kid)
                    elif metric == "nr. sessions":
                        val = col.number_input(metric, 0, 10, 1, key=kid)
                    else:
                        val = col.number_input(metric, 0.0, 200.0, 0.0, 0.5, key=kid)
                    input_data[f"{metric}" + (f".{d_idx}" if d_idx > 0 else "")] = float(val)

        run = st.button("Calculer le risque", type="primary")

    with col_result:
        st.markdown(panel(title="Résultat de la prédiction"), unsafe_allow_html=True)

        if run:
            row = {f: input_data.get(f, 0.0) for f in feat_names}
            rdf = add_aggregate_features(pd.DataFrame([row]))
            for c in feat_names:
                if c not in rdf.columns:
                    rdf[c] = 0.0
            rf = rdf[list(feat_names)]

            p_val = float(models[sel_model].predict_proba(rf)[0, 1])
            t_val = _thr(sel_model)
            is_h  = p_val >= t_val

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_val * 100,
                number={"suffix": "%",
                        "font": {"size": 40, "family": "Inter",
                                  "color": "#dc2626" if is_h else "#059669"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8",
                              "tickfont": {"size": 10, "family": "Inter"}},
                    "bar":  {"color": "#dc2626" if is_h else "#059669", "thickness": 0.22},
                    "bgcolor": "#f8fafc", "borderwidth": 0,
                    "steps": [
                        {"range": [0, t_val * 100],  "color": "#f0fdf4"},
                        {"range": [t_val * 100, 100], "color": "#fef2f2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 2},
                        "thickness": 0.7, "value": t_val * 100,
                    },
                },
                title={
                    "text": f"Score de risque · seuil {t_val:.0%}",
                    "font": {"size": 13, "color": "#374151", "family": "Inter"},
                },
            ))
            gauge.update_layout(
                height=260, margin=dict(l=16, r=16, t=60, b=8),
                paper_bgcolor="#fff", font=dict(family="Inter"),
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(risk_banner(is_h, p_val, t_val), unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Probabilité", f"{p_val:.1%}")
            m2.metric("Seuil d'alerte", f"{t_val:.2f}")
        else:
            st.markdown(empty_state(
                "Renseignez les données d'entraînement et cliquez sur Calculer le risque."
            ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Interprétabilité SHAP":
    st.markdown(page_title(
        "Interprétabilité SHAP",
        "Contribution de chaque feature à la décision du modèle"
    ), unsafe_allow_html=True)

    st.markdown("""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;
     border-radius:8px;padding:14px 20px;margin-bottom:24px;font-size:13px;
     color:#1e40af;line-height:1.7">
  <strong>SHAP</strong> (SHapley Additive exPlanations) — une valeur SHAP positive
  indique que la feature <em>augmente</em> la probabilité de blessure, une valeur
  négative la <em>réduit</em>. L'amplitude absolue mesure l'importance sur la décision.
</div>""", unsafe_allow_html=True)

    for lbl, bar_f, bee_f, csv_f in [
        ("Random Forest (modèle recommandé)",
         "shap_summary_Random_Forest.png", "shap_beeswarm_Random_Forest.png",
         "shap_top_features_rf.csv"),
        ("XGBoost",
         "shap_summary_XGBoost.png", "shap_beeswarm_XGBoost.png",
         "shap_top_features_xgb.csv"),
    ]:
        csv_p = os.path.join(ROOT, "results", csv_f)
        bar_p = os.path.join(ROOT, "figures", bar_f)
        bee_p = os.path.join(ROOT, "figures", bee_f)

        with st.expander(lbl, expanded=(lbl.startswith("Random"))):
            # Plotly bar chart from CSV (primary visualisation)
            if os.path.exists(csv_p):
                shap_df = pd.read_csv(csv_p).head(20)
                fig_s = px.bar(
                    shap_df, x="mean_abs_shap", y="feature", orientation="h",
                    labels={"mean_abs_shap": "SHAP moyen |absolu|", "feature": ""},
                    color="mean_abs_shap",
                    color_continuous_scale=[[0, "#dbeafe"], [1, "#2563eb"]],
                )
                _fig(fig_s, "Top 20 features les plus influentes", 500)
                fig_s.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    margin=dict(l=180, r=20, t=44, b=36),
                )
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.warning(
                    f"CSV SHAP non disponible pour {lbl}. "
                    "Relancez : python train.py"
                )

            # Static images as supplementary detail
            if os.path.exists(bar_p) and os.path.exists(bee_p):
                img1, img2 = st.columns(2, gap="medium")
                with img1:
                    st.markdown("""
<div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;
     letter-spacing:.06em;margin-bottom:8px">Importance globale (mean |SHAP|)</div>""",
                                unsafe_allow_html=True)
                    st.image(bar_p, use_container_width=True)
                with img2:
                    st.markdown("""
<div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;
     letter-spacing:.06em;margin-bottom:8px">Beeswarm — impact individuel</div>""",
                                unsafe_allow_html=True)
                    st.image(bee_p, use_container_width=True)

    # Legend
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;padding:20px 22px">
  <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;
       letter-spacing:.08em;margin-bottom:16px">Guide de lecture</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
    <div style="background:#fef2f2;border-radius:6px;padding:14px;border-left:3px solid #dc2626">
      <div style="font-size:12px;font-weight:600;color:#dc2626;margin-bottom:6px">
        Valeur SHAP positive</div>
      <div style="font-size:12px;color:#7f1d1d;line-height:1.6">
        La feature augmente le risque prédit.</div>
    </div>
    <div style="background:#f0fdf4;border-radius:6px;padding:14px;border-left:3px solid #059669">
      <div style="font-size:12px;font-weight:600;color:#059669;margin-bottom:6px">
        Valeur SHAP négative</div>
      <div style="font-size:12px;color:#14532d;line-height:1.6">
        La feature réduit le risque prédit.</div>
    </div>
    <div style="background:#eff6ff;border-radius:6px;padding:14px;border-left:3px solid #2563eb">
      <div style="font-size:12px;font-weight:600;color:#2563eb;margin-bottom:6px">
        Amplitude absolue élevée</div>
      <div style="font-size:12px;color:#1e40af;line-height:1.6">
        Feature très influente sur la décision.</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulation":
    st.markdown(page_title(
        "Simulation — Analyse de scénarios",
        "Impact d'une réduction de charge d'entraînement sur le risque de blessure"
    ), unsafe_allow_html=True)

    p_left, p_right = st.columns([1, 2], gap="large")

    with p_left:
        st.markdown(panel(title="Profil du coureur"), unsafe_allow_html=True)
        sel_sim  = st.selectbox(
            "Modèle",
            list(models.keys()),
            format_func=lambda k: LABELS[k] + (" (recommandé)" if k == "random_forest" else ""),
            key="sim_key",
        )
        base_km  = st.slider("Volume hebdomadaire (km)", 0.0, 150.0, 60.0, 5.0)
        base_ses = st.slider("Sessions / semaine", 1, 14, 6)
        base_int = st.slider("Km haute intensité (Z5-T1-T2)", 0.0, 30.0, 5.0, 0.5)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        red_pct  = st.slider("Réduction de charge appliquée (%)", 0, 60, 0, 5)
        go_sim   = st.button("Lancer la simulation", type="primary")

    def _sim_row(km_f, int_f):
        row = {}
        for feat in feat_names:
            if feat.startswith("agg_") or feat == "acwr_total_km":
                row[feat] = 0.0
            elif "total km"      in feat: row[feat] = base_km  / 7 * km_f
            elif "nr. sessions"  in feat: row[feat] = base_ses / 7
            elif "km Z5-T1-T2"   in feat: row[feat] = base_int / 7 * int_f
            elif "km Z3-4"       in feat: row[feat] = base_int / 7 * 0.5 * int_f
            elif "km sprinting"  in feat: row[feat] = base_km  / 7 * 0.1 * km_f
            elif "perceived"     in feat: row[feat] = max(0.05, 0.3 * km_f + 0.1)
            else:                         row[feat] = 0.0
        rdf = add_aggregate_features(pd.DataFrame([row]))
        for c in feat_names:
            if c not in rdf.columns:
                rdf[c] = 0.0
        return rdf[list(feat_names)]

    with p_right:
        if go_sim:
            t_sim  = _thr(sel_sim)
            reds   = list(range(0, 65, 5))
            probs_ = [
                float(models[sel_sim].predict_proba(
                    _sim_row(1 - r / 100, 1 - r / 100))[0, 1])
                for r in reds
            ]
            cur_p  = probs_[red_pct // 5]

            fig_sim = go.Figure()
            y_max = max(max(probs_) * 100 + 5, t_sim * 100 + 8)

            # Risk zones background
            fig_sim.add_hrect(y0=0, y1=t_sim * 100,
                               fillcolor="#f0fdf4", opacity=0.5, line_width=0)
            fig_sim.add_hrect(y0=t_sim * 100, y1=y_max,
                               fillcolor="#fef2f2", opacity=0.5, line_width=0)

            # Threshold line
            fig_sim.add_hline(
                y=t_sim * 100, line_dash="dash",
                line_color="#d97706", line_width=1.5,
                annotation_text=f"Seuil ({t_sim:.0%})",
                annotation_position="top right",
                annotation_font=dict(size=11, color="#92400e"),
            )

            # Risk curve
            fig_sim.add_trace(go.Scatter(
                x=reds, y=[p * 100 for p in probs_],
                mode="lines+markers", name="Probabilité blessure",
                line=dict(color="#2563eb", width=2.5),
                marker=dict(
                    size=8,
                    color=["#dc2626" if p >= t_sim else "#059669" for p in probs_],
                    line=dict(color="#fff", width=1.5),
                ),
                hovertemplate="<b>Réduction %{x}%</b><br>Risque : %{y:.1f}%<extra></extra>",
            ))

            # Current selected point
            fig_sim.add_trace(go.Scatter(
                x=[red_pct], y=[cur_p * 100],
                mode="markers", name="Scénario sélectionné",
                marker=dict(size=14, color="#2563eb",
                            line=dict(color="#fff", width=2.5)),
            ))

            _fig(fig_sim, "Impact d'une réduction de charge sur le risque de blessure")
            fig_sim.update_layout(
                xaxis_title="Réduction appliquée (%)",
                yaxis_title="Probabilité de blessure (%)",
                yaxis_range=[0, y_max], height=380,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Summary metrics
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Risque calculé", f"{cur_p:.1%}",
                        delta=f"{(probs_[0] - cur_p) * 100:+.1f}% vs base")
            mc2.metric("Volume de référence", f"{base_km:.0f} km/sem")
            mc3.metric("Réduction", f"-{red_pct}%")

            # Verdict banner
            safe = next((r for r, p in zip(reds, probs_) if p < t_sim), None)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if cur_p < t_sim:
                st.markdown(f"""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #059669;
     border-radius:8px;padding:16px 20px;font-size:13px;color:#166534;line-height:1.7">
  <strong>Charge sous le seuil d'alerte</strong> avec -{red_pct}% de volume.
  Probabilité : {cur_p:.1%} &lt; {t_sim:.0%}.
</div>""", unsafe_allow_html=True)
            elif safe is not None:
                st.markdown(f"""
<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;
     border-radius:8px;padding:16px 20px;font-size:13px;color:#92400e;line-height:1.7">
  Une réduction de <strong>{safe}%</strong> ramènerait le risque sous le seuil
  ({t_sim:.0%}). Actuellement à {cur_p:.1%}.
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div style="background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;
     border-radius:8px;padding:16px 20px;font-size:13px;color:#991b1b;line-height:1.7">
  La réduction de charge seule ne suffit pas. Envisager de réduire également
  l'intensité et d'allonger les périodes de récupération.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(empty_state(
                "Configurez le profil du coureur et lancez la simulation."
            ), unsafe_allow_html=True)
