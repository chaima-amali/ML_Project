# 🎓 Student Learning Profile Clustering
### ENSIA · Machine Learning Module · Spring 2025–2026

> An end-to-end unsupervised machine learning system that identifies meaningful learning behaviour profiles among university students, and delivers personalised academic recommendations through an interactive dashboard.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notebooks](#-notebooks)
- [Dashboard](#-dashboard)
- [Model Summary](#-model-summary)
- [Student Profiles](#-student-profiles)
- [Team](#-team)

---

## 🔍 Project Overview

This project applies unsupervised machine learning to a dataset of **12,469 anonymised university student records** across 10 behavioural and academic features. The objective is to cluster students into interpretable learning profiles and surface actionable, personalised recommendations.

**Three clustering algorithms** are implemented and compared:

| Algorithm | Approach | Key Advantage |
|---|---|---|
| **K-Means** (signal-aware) | Partition-based | Highest Silhouette Score (≈0.998) |
| **DBSCAN** | Density-based | Built-in noise/outlier detection |
| **Hierarchical** | Agglomerative (Ward) | Dendrogram-based interpretability |

The project culminates in a **Streamlit dashboard** where any student can input their academic behaviour and receive their learning profile assignment with personalised recommendations.

---

## ✨ Features

- **Signal-Aware K-Means** — amplifies discriminative binary features to recover natural cluster structure
- **Three-Algorithm Comparison** — K-Means, DBSCAN, and Hierarchical evaluated on standard internal metrics
- **Interactive Dashboard** — real-time profile prediction, radar chart, match confidence bars, and recommendations
- **Professional Notebooks** — five fully documented Jupyter notebooks covering the complete ML pipeline
- **Reproducible Pipeline** — all paths use `pathlib` relative references; runs from any environment

---

## 🛠️ Technologies Used

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn` (KMeans, DBSCAN, AgglomerativeClustering, PCA, t-SNE) |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Statistics | `scipy` (linkage, cophenetic, chi2_contingency) |
| Notebooks | `jupyter`, `nbformat` |

**Python version:** 3.10+

---

## 📁 Project Structure

```
project_final/
│
├── Data/
│   ├── raw/
│   │   └── merged_dataset.csv              ← Original dataset (14,003 rows, 16 features)
│   └── processed/
│       ├── cleaned_dataset.csv             ← After deduplication & normalisation (12,469 rows)
│       ├── kmeans_clustered.csv            ← With KMeans_Cluster + DBSCAN_Cluster labels
│       ├── hierarchical_labelled_dataset.csv ← With Hierarchical_Cluster labels
│       └── dbscan_labelled_dataset.csv     ← With DBSCAN_Cluster labels
│
├── notebooks/
│   ├── data_collection.ipynb              ← NB1: Data loading, cleaning & normalisation
│   ├── eda.ipynb                          ← NB2: Exploratory data analysis
│   ├── kmeans_dbscan.ipynb               ← NB3: K-Means & DBSCAN clustering
│   ├── hierarchical_clustering.ipynb      ← NB4: Hierarchical clustering & dendrograms
│   └── evaluation.ipynb                  ← NB5: Three-algorithm comparison & profiles
│
├── dashboard/
│   ├── app.py                             ← Streamlit application (fixed & enhanced)
│   └── requirements.txt                  ← Python dependencies
│
└── README.md                              ← This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1 — Clone / Extract the project

```bash
# If using git:
git clone <repository-url>
cd project_final

# Or extract the ZIP and navigate into it:
cd project_final
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r dashboard/requirements.txt
```

For running the Jupyter notebooks, also install:

```bash
pip install jupyter nbformat ipykernel
```

---

## 📖 Usage

### Running the Dashboard

```bash
cd dashboard
streamlit run app.py
```

Streamlit will print a local URL (e.g., `http://localhost:8501`). Open it in any browser.

> **Note:** The app automatically resolves the dataset path relative to `app.py`.  
> Keep the project folder structure intact.

### Running the Notebooks

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Navigate to the `notebooks/` folder.
3. Run notebooks **in order** (1 → 5). Each notebook reads from the output of the previous one.
4. Use **Kernel → Restart & Run All** for a clean run.

**Recommended run order:**

```
data_collection.ipynb → eda.ipynb → kmeans_dbscan.ipynb
                                  → hierarchical_clustering.ipynb
                                  → evaluation.ipynb
```

---

## 📓 Notebooks

| # | Notebook | Description | Author Role |
|---|---|---|---|
| 1 | `data_collection.ipynb` | Raw data loading, quality checks, deduplication, Min-Max normalisation | Data Engineer |
| 2 | `eda.ipynb` | Feature distributions, outlier detection, correlation analysis, signal ranking | EDA Specialist |
| 3 | `kmeans_dbscan.ipynb` | Signal-aware K-Means + DBSCAN, elbow/silhouette selection, cluster profiling | Clustering Engineer |
| 4 | `hierarchical_clustering.ipynb` | Linkage comparison (CCC), dendrogram analysis, Agglomerative Clustering | ML Specialist |
| 5 | `evaluation.ipynb` | Three-algorithm comparison, PCA/t-SNE visualisations, profile heatmaps, recommendations | Evaluations Lead |

Each notebook contains:
- A professional header banner
- A notebook report table (Objective / Methodology / Results / Conclusion)
- Clean, well-commented code cells
- Academic-quality visualisations

---

## 🖥️ Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides:

- **Profile Prediction** — Enter your academic behaviour values via sliders and dropdowns; receive an instant cluster assignment
- **Radar Chart** — Compare your profile to the cluster centroid across 8 key dimensions
- **Confidence Bars** — See your match strength against all 4 profiles
- **Personalised Recommendations** — Tailored, actionable advice for your learning archetype
- **Profile Overview** — Reference cards for all 4 learning profiles
- **Quick-Set Presets** — Load typical values for any profile to explore the model

### Dashboard Fixes Applied

The original `app.py` contained the following issues, all resolved:

| Issue | Fix |
|---|---|
| `load_model` missing `return` statement | Added complete return tuple |
| Duplicate code blocks (results rendered twice) | Removed duplicate section |
| Syntax error in `cols[i % 2=` | Fixed to `cols[i % 2]` |
| Incorrect relative data path | Replaced with `pathlib`-based absolute resolution |
| `predict_cluster` function called before definition | Moved all functions before usage |
| Inconsistent profile count (4 vs 6) | Unified to 4 throughout |

---

## 📊 Model Summary

| Property | Value |
|---|---|
| Algorithm | K-Means (k=4, k-means++, n_init=30) |
| Dataset (clean) | 12,469 students |
| Features | 10 behavioural & academic features |
| Preprocessing | MinMaxScaler → Feature Amplification → PCA (≥90% variance) |
| Silhouette Score | **≈ 0.998** |
| Davies-Bouldin Index | Low (well-separated clusters) |

### Why Signal-Aware Weighting?

Standard scaling produced a Silhouette Score of only ~0.12 because the binary features (`Extracurricular`, `Discussions`) were overwhelmed by continuous noise. Feature amplification (×5 for binary signals) recovers the natural cluster structure defined by the 2×2 combinations of participation behaviour.

---

## 👥 Student Profiles

| # | Profile | Size | Key Signal |
|---|---|---|---|
| 0 | 😴 Disengaged | ~17% | Extracurricular: No · Discussions: No |
| 1 | 🌟 Fully Engaged | ~35% | Extracurricular: Yes · Discussions: Yes |
| 2 | 🏃 Socially Active | ~23% | Extracurricular: Yes · Discussions: No |
| 3 | 📚 Academic-Focused | ~25% | Extracurricular: No · Discussions: Yes |

---

## 👨‍💻 Team

| Role | Responsibilities |
|---|---|
| Member 1 — Data Engineer | Data loading, cleaning, normalisation (Notebook 1) |
| Member 2 — EDA Specialist | Exploratory analysis, feature engineering (Notebook 2) |
| Member 3 — Clustering Engineer | K-Means & DBSCAN implementation (Notebook 3) |
| Member 4 — ML Specialist | Hierarchical clustering & dendrograms (Notebook 4) |
| Member 5 — Evaluations Lead | Three-algorithm evaluation & dashboard (Notebook 5) |

---

*ENSIA · Machine Learning Module · Spring 2025–2026*
