# 🎓 Learning Behavior Clustering & Student Profiling
**ENSIA — Machine Learning Module | Spring 2025-2026**

An end-to-end unsupervised machine learning system that analyzes anonymized student academic and behavioral data to automatically discover meaningful learning profiles such as High Performers, Last-Minute Learners, Passive Students, and At-Risk Students.

---

## 📁 Project Structure
ML_Project/
├── data/
│   ├── raw/                    ← original/simulated data files
│   └── processed/              ← cleaned and normalized data
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_evaluation.ipynb
├── models/                     ← saved trained models (.pkl files)
├── figures/                    ← all generated plots and charts
├── dashboard.py                ← optional Streamlit dashboard
├── requirements.txt
└── README.md

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/chaima-amali/ML_Project.git
cd ML_Project
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch Jupyter**
```bash
jupyter notebook
```

---

## 🚀 How to Run

Run the notebooks **in order**:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_collection.ipynb` | Load or simulate student data |
| 2 | `02_preprocessing.ipynb` | Clean, normalize, handle missing values |
| 3 | `03_eda.ipynb` | Exploratory analysis and visualizations |
| 4 | `04_feature_engineering.ipynb` | Create derived features |
| 5 | `05_clustering.ipynb` | Train K-Means, DBSCAN, Hierarchical models |
| 6 | `06_evaluation.ipynb` | Metrics, PCA/t-SNE, cluster interpretation |

**Optional — Run the dashboard:**
```bash
streamlit run dashboard.py
```

---

## 🤖 Algorithms Used

- **K-Means** — Primary model. Elbow method used to select optimal K.
- **DBSCAN** — Density-based. Detects outlier students automatically.
- **Hierarchical Clustering** — Agglomerative with Ward linkage. Visualized via dendrogram.

---

## 📏 Evaluation Metrics

| Metric | Goal |
|--------|------|
| Silhouette Score | Closer to +1 is better |
| Davies-Bouldin Index | Lower is better |
| Calinski-Harabasz Score | Higher is better |
| PCA / t-SNE Visualization | Visually well-separated clusters |

---

## 🏷️ Student Profiles Identified

| Profile | Characteristics |
|---------|----------------|
| 🌟 High Performers | High GPA, high attendance, submits on time, active on LMS |
| ⏰ Last-Minute Learners | Average GPA, high submission delay, low LMS activity |
| 😴 Passive Students | Average GPA, low attendance, rarely engages |
| ⚠️ At-Risk Students | Low GPA, poor attendance, misses deadlines, no LMS activity |


## 📌 Notes

- All student data is fully anonymized — no real personal information is used.
- This project is for academic purposes only (ENSIA ML Module).
- Models are saved using `joblib` and can be reloaded without retraining.