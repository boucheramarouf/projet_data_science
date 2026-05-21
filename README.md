# RunSafe AI — Injury Prediction for Competitive Runners

**RNCP40875 — Expert en Ingénierie de Données · Bloc 2 · EFREI 2025-26**

| | |
|---|---|
| **Étudiantes** | Bouchera MAROUF · Manel TOUATI |
| **Encadrante** | Sarah Malaeb |
| **Type** | Classification supervisée — déséquilibre de classes fort (~1.4 %) |
| **Tâche** | Prédire le risque de blessure d'un coureur pour la semaine suivante |

---

## Problématique

À partir de la **charge d'entraînement des 7 derniers jours** (volume, intensité, récupération perçue…), peut-on prédire si un coureur compétitif va se blesser la semaine suivante ?

Cas d'usage métier : un staff médical ou un entraîneur peut saisir les données hebdomadaires d'un athlète et obtenir immédiatement un score de risque avec recommandations d'action.

---

## Dataset

- **Source :** Kaggle — "Sports Injury Prediction" (fenêtres glissantes 7 jours)
- **42 766 observations** · 73 colonnes · taux de blessure ≈ 1.4 %
- **10 métriques quotidiennes × 7 jours :** volume (km total, zones d'intensité, sprint), force, heures alternatives, métriques perçues (exertion, succès d'entraînement, récupération)
- **Variable cible :** `injury` (0 = non blessé, 1 = blessé la semaine suivante)

---

## Structure du projet

```
projet_2/
├── data/
│   └── raw/          # Dataset brut CSV
├── src/
│   ├── preprocessing.py   # Nettoyage, feature engineering, pipeline sklearn
│   ├── models.py          # 5 modèles (LR, RF, XGB, SVM, MLP)
│   └── evaluation.py      # Métriques, SHAP, cross-validation
├── dashboard/
│   └── app.py             # Dashboard Streamlit (5 pages)
├── api/
│   ├── main.py            # API FastAPI (/predict, /health, /model-info)
│   └── schemas.py         # Schémas Pydantic
├── notebooks/
│   └── 01_EDA_and_modeling.ipynb  # EDA complète + expérimentation
├── figures/               # Courbes ROC/PR, matrices de confusion, SHAP
├── results/               # CSV comparatif, seuils, features SHAP
├── saved_models/          # Modèles sérialisés (.pkl)
├── train.py               # Script d'entraînement principal
├── run_cv.py              # Validation croisée 5-Fold (séparé)
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1 — Entraîner les modèles

```bash
python train.py
```

Génère dans `saved_models/`, `figures/` et `results/`.

### 2 — Lancer la validation croisée (optionnel, ~15 min)

```bash
python run_cv.py
```

Génère `results/cv_results.csv` (affiché dans le dashboard, onglet Comparaison).

### 3 — Démarrer le dashboard Streamlit

```bash
python -m streamlit run dashboard/app.py
```

Accès : [http://localhost:8501](http://localhost:8501)

### 4 — Démarrer l'API REST (optionnel)

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Modèles implémentés

| Modèle | Type | Gestion déséquilibre |
|---|---|---|
| Logistic Regression | Baseline linéaire | `class_weight="balanced"` |
| Random Forest | Ensemble (arbres) | `class_weight="balanced"` |
| XGBoost | Gradient boosting | `scale_pos_weight` |
| SVM (LinearSVC calibré) | Noyau linéaire | `class_weight="balanced"` |
| **MLP (Deep Learning)** | Réseau 256-128-64 | `sample_weight` par classe |

---

## Résultats (jeu de test, 20 %)

| Modèle | F1 (blessé) | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| **Random Forest ** | **0.076** | **0.188** | **0.032** | **0.681** |
| XGBoost | 0.070 | 0.111 | 0.028 | 0.637 |
| Logistic Regression | 0.060 | 0.402 | 0.028 | 0.690 |
| MLP (Deep Learning) | 0.052 | 0.171 | 0.024 | 0.646 |
| SVM | 0.016 | 0.009 | 0.029 | 0.690 |

> Les faibles valeurs absolues de F1 et PR-AUC sont attendues avec 1.4 % de blessures. La PR-AUC est la métrique principale (elle est 2× à 5× supérieure à la baseline aléatoire de 0.014).

**Modèle recommandé : Random Forest** — meilleur compromis PR-AUC, ROC-AUC, stabilité et interprétabilité (SHAP TreeExplainer).

---

## Dashboard — Pages

| Page | Contenu |
|---|---|
| Accueil | KPIs, distribution des classes, corrélations |
| Comparaison | Tableau comparatif, radar chart, PR/ROC curves, matrices de confusion, CV |
| Prédiction | Saisie 7 jours, jauge de risque, alerte avec recommandation |
| SHAP | Feature importance globale (bar) + individuelle (beeswarm) |
| Simulation | Courbe de risque selon réduction de charge (What-If) |

---

## API REST — Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Vérification du service et des modèles chargés |
| `GET` | `/model-info` | Liste des modèles disponibles et leurs seuils |
| `POST` | `/predict` | Prédiction du risque (JSON 7 jours → probabilité + niveau de risque) |

**Exemple de requête `/predict` :**
```json
{
  "day_0": {"total_km": 10.5, "perceived_recovery": 0.6},
  "model": "random_forest"
}
```

---

## Compétences RNCP40875 — Bloc 2 validées

- C3.1 Préparation et transformation des données
- C3.2 Dashboard décisionnel interactif
- C3.3 Analyse exploratoire et insights métier
- C4.2 Développement de modèles prédictifs ML/DL
- C4.3 Évaluation comparative multi-modèles
