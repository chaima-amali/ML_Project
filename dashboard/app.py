"""
Student Learning Profile Predictor — Streamlit Dashboard
=========================================================
ENSIA · Machine Learning Module · 2025–2026

Run with:
    cd dashboard
    streamlit run app.py

Requires:
    pip install -r requirements.txt
"""

# ─── Standard imports ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ─── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Learning Profile Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).parent
DATA_PATH     = DASHBOARD_DIR.parent / "Data" / "raw" / "merged_dataset.csv"

# ─── Feature configuration ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "StudyHours", "Attendance", "Extracurricular", "Motivation",
    "OnlineCourses", "Discussions", "AssignmentCompletion",
    "ExamScore", "StressLevel", "FinalGrade",
]

RADAR_LABELS = [
    "StudyHours", "Attendance", "AssignmentCompletion", "ExamScore",
    "Motivation", "StressLevel", "Extracurricular", "Discussions",
]

FEATURE_WEIGHTS = {
    "Extracurricular":     5.0,
    "Discussions":         5.0,
    "Motivation":          2.0,
    "StudyHours":          1.0,
    "Attendance":          1.0,
    "OnlineCourses":       1.0,
    "AssignmentCompletion":1.0,
    "ExamScore":           1.0,
    "StressLevel":         1.0,
    "FinalGrade":          1.0,
}

# ─── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    0: {"hex": "#2E86AB", "light": "#EBF5FB", "name": "Blue"},   # Disengaged
    1: {"hex": "#E94560", "light": "#FDEDEE", "name": "Red"},    # Fully Engaged
    2: {"hex": "#06A77D", "light": "#E8F8F5", "name": "Green"},  # Socially Active
    3: {"hex": "#F18F01", "light": "#FEF9E7", "name": "Amber"},  # Academic-Focused
}

# ─── Profile definitions ───────────────────────────────────────────────────────
PROFILES = {
    0: {
        "name":   "😴 Disengaged",
        "short":  "Disengaged",
        "pct":    "~17%",
        "desc":   (
            "This profile shows lower-than-average engagement across most metrics, "
            "including attendance and assignment completion. There is significant "
            "potential for growth with targeted support and early intervention."
        ),
        "traits": ["Low engagement", "Below-average scores", "High growth potential"],
        "key_signals": ["Extracurricular: No", "Discussions: No"],
        "recs": [
            ("🚪", "Speak to your academic advisor this week — early outreach can make a significant difference."),
            ("📋", "Use a weekly task list to build momentum, starting with small, achievable goals."),
            ("💻", "Leverage digital resources like recorded lectures and online tutorials to catch up."),
            ("❤️",  "Address any wellbeing barriers first; academic progress often follows personal stability."),
        ],
    },
    1: {
        "name":   "🌟 Fully Engaged",
        "short":  "Fully Engaged",
        "pct":    "~35%",
        "desc":   (
            "A top-performing profile with high participation in both academic and "
            "extracurricular activities. You are a well-rounded, highly motivated "
            "student with strong academic outcomes."
        ),
        "traits": ["High participation", "Strong academic scores", "Well-rounded"],
        "key_signals": ["Extracurricular: Yes", "Discussions: Yes"],
        "recs": [
            ("🏆", "Apply for research assistantships, competitions, or scholarships to leverage your strengths."),
            ("🧑‍🏫", "Mentor peers or lead study groups — teaching is the deepest form of mastery."),
            ("💡", "Build a portfolio of projects, contributions, or publications to showcase your skills."),
            ("🔭", "Explore advanced electives or independent study to push beyond the standard curriculum."),
        ],
    },
    2: {
        "name":   "🏃 Socially Active",
        "short":  "Socially Active",
        "pct":    "~23%",
        "desc":   (
            "This profile is characterized by high involvement in extracurricular "
            "activities but lower participation in academic discussions. You are "
            "socially integrated but could improve your academic engagement."
        ),
        "traits": ["High extracurriculars", "Lower academic discussion", "Socially integrated"],
        "key_signals": ["Extracurricular: Yes", "Discussions: No"],
        "recs": [
            ("💬", "Connect your social skills to academics — join or form a structured study group."),
            ("🎯", "Set a goal to contribute meaningfully in at least one academic discussion per week."),
            ("⚖️",  "Balance your time to ensure social activities support, rather than detract from, studies."),
            ("🤝", "Use your social network to find academic mentors or collaborative project partners."),
        ],
    },
    3: {
        "name":   "📚 Academic-Focused",
        "short":  "Academic-Focused",
        "pct":    "~25%",
        "desc":   (
            "You are highly engaged in academic discussions but participate less "
            "in extracurricular activities. You have a strong academic focus and "
            "could benefit from broader campus involvement to develop holistically."
        ),
        "traits": ["High academic discussion", "Lower extracurriculars", "Academically focused"],
        "key_signals": ["Extracurricular: No", "Discussions: Yes"],
        "recs": [
            ("🎨", "Join one non-academic club or society to broaden your horizons and prevent burnout."),
            ("👥", "Your academic insights are valuable — share them in team-based and collaborative projects."),
            ("🌐", "Attend campus events or workshops to build a wider professional and social network."),
            ("🧘", "Ensure you take regular breaks and engage in hobbies or activities outside your studies."),
        ],
    },
}

# ─── Centroid reference profiles ──────────────────────────────────────────────
# Derived from K-Means centroids in the Kmeans_DBSCAN_improved notebook.
CENTROIDS_DF = pd.DataFrame({
    "StudyHours":          [16.35, 34.54, 16.31, 34.56],
    "Attendance":          [75.41, 89.73, 75.39, 89.81],
    "Extracurricular":     [0.00,  1.00,  1.00,  0.00],
    "Motivation":          [1.00,  1.00,  1.00,  1.00],
    "OnlineCourses":       [9.83,  9.99,  9.81,  9.98],
    "Discussions":         [0.00,  1.00,  0.00,  1.00],
    "AssignmentCompletion":[74.81, 85.06, 74.75, 85.11],
    "ExamScore":           [69.89, 80.13, 69.88, 80.17],
    "StressLevel":         [1.00,  1.00,  1.00,  1.00],
    "FinalGrade":          [1.50,  1.00,  1.50,  1.00],
})

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4               { font-family: 'Space Grotesk', sans-serif; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B4F72 55%, #117A65 100%);
    border-radius: 16px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.6rem;
    overflow: hidden;
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2rem;
    margin: 0 0 0.4rem;
    font-weight: 700;
}
.hero-banner p { color: #AED6F1; margin: 0; font-size: 1rem; line-height: 1.6; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 3px 13px;
    font-size: 0.77rem;
    color: #ffffff;
    margin: 8px 4px 0 0;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── Profile result card ── */
.profile-card {
    border-radius: 14px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1rem;
    border: 1.5px solid;
}
.profile-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    margin-bottom: 3px;
}
.profile-subtitle { font-size: 0.83rem; opacity: 0.75; margin-bottom: 10px; }
.profile-desc     { font-size: 0.93rem; line-height: 1.68; opacity: 0.9; }

/* ── Metric card ── */
.metric-box {
    background: #F8FAFC;
    border-radius: 10px;
    padding: 0.9rem 0.5rem;
    text-align: center;
    border: 0.5px solid #E2E8F0;
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 600;
    margin-bottom: 2px;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Recommendation item ── */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.7rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 7px;
    background: #F8FAFC;
    font-size: 0.9rem;
    line-height: 1.55;
    border-left: 3px solid;
}

/* ── Section header ── */
.section-hdr {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748B;
    margin: 1.3rem 0 0.65rem;
    padding-bottom: 5px;
    border-bottom: 1px solid #E2E8F0;
}

/* ── Trait pill ── */
.trait-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 13px;
    font-size: 0.77rem;
    font-weight: 500;
    margin: 4px 4px 0 0;
    border: 1px solid;
}

/* ── Overview card ── */
.overview-card {
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    border-top-width: 3px;
    border-top-style: solid;
    border-left: 1px solid;
    border-right: 1px solid;
    border-bottom: 1px solid;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div { padding-top: 0.8rem; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Model loading & prediction (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Training K-Means on student data…")
def load_model(data_path: str):
    """
    Load the dataset, apply the signal-aware preprocessing pipeline, and train K-Means.

    Pipeline:
        1. MinMaxScaler  → scale all features to [0, 1]
        2. Feature Amplification → weight binary/ordinal signals
        3. PCA (≥90% variance)   → compress to dominant components
        4. KMeans (k=4, k-means++) → fit cluster assignments

    Returns
    -------
    df        : pd.DataFrame  — cleaned source data
    scaler    : MinMaxScaler  — fitted scaler
    weights   : dict          — feature weight map
    pca       : PCA           — fitted PCA
    kmeans    : KMeans        — fitted model
    """
    df = pd.read_csv(data_path)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median())

    # Step 1 — MinMax scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    # Step 2 — Feature amplification
    for feat, w in FEATURE_WEIGHTS.items():
        X_scaled[feat] = X_scaled[feat] * w
    X_weighted = X_scaled.values

    # Step 3 — PCA (retain ≥90% variance)
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_weighted)

    # Step 4 — K-Means
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=30, random_state=42)
    kmeans.fit(X_pca)

    return df, scaler, FEATURE_WEIGHTS, pca, kmeans


def predict_cluster(inputs: dict, scaler, weights: dict, pca, kmeans):
    """
    Assign a student to a cluster and return distances to all centroids.

    Parameters
    ----------
    inputs : dict  — raw feature values keyed by FEATURE_COLS
    Returns
    -------
    cluster_id : int
    distances  : np.ndarray shape (4,)
    """
    row = pd.DataFrame([inputs], columns=FEATURE_COLS)
    row_scaled = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLS)

    for feat, w in weights.items():
        row_scaled[feat] = row_scaled[feat] * w

    row_pca = pca.transform(row_scaled.values)
    cluster_id = int(kmeans.predict(row_pca)[0])

    # Euclidean distances to every centroid
    diffs = kmeans.cluster_centers_ - row_pca
    distances = np.linalg.norm(diffs, axis=1)
    return cluster_id, distances


# ─── Chart helpers ─────────────────────────────────────────────────────────────
def make_radar(user_vals: list, centroid_vals: list, cluster_id: int) -> go.Figure:
    """Dual-trace polar chart comparing user to cluster centroid."""
    col = COLORS[cluster_id]["hex"]
    r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
    fill_col = f"rgba({r},{g},{b},0.15)"

    labels = RADAR_LABELS + [RADAR_LABELS[0]]           # close the polygon
    user_vals      = user_vals      + [user_vals[0]]
    centroid_vals  = centroid_vals  + [centroid_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=centroid_vals, theta=labels, name="Cluster average",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
        fill="toself", fillcolor="rgba(148,163,184,0.10)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_vals, theta=labels, name="Your profile",
        line=dict(color=col, width=2.5),
        fill="toself", fillcolor=fill_col,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            showticklabels=False, gridcolor="#E2E8F0"),
            angularaxis=dict(tickfont=dict(size=11, family="DM Sans")),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=50, r=50, t=20, b=50),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_confidence_bar(distances: np.ndarray) -> go.Figure:
    """Horizontal bar chart showing match confidence for each profile."""
    max_d  = distances.max() or 1.0
    confs  = [max(0, round((1 - d / max_d) * 100)) for d in distances]
    order  = list(np.argsort(distances))           # best → worst match

    names  = [PROFILES[i]["short"]  for i in order]
    values = [confs[i]              for i in order]
    colors = [COLORS[i]["hex"]      for i in order]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 118], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11, family="DM Sans")),
        margin=dict(l=10, r=55, t=10, b=10),
        height=210,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def normalize_for_radar(inputs: dict, centroid_row: pd.Series) -> tuple:
    """Scale user inputs and centroid to [0,1] for radar display."""
    combined = pd.DataFrame(
        [inputs, centroid_row.to_dict()],
        columns=RADAR_LABELS
    )
    norm = MinMaxScaler().fit_transform(combined)
    return norm[0].tolist(), norm[1].tolist()


# ─── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🎓 Student Learning Profile Predictor</h1>
  <p>
    K-Means clustering trained on 12,469 anonymised student records —
    discover your learning archetype and receive personalised recommendations.
  </p>
  <span class="hero-badge">K-Means · k = 4</span>
  <span class="hero-badge">12,469 students</span>
  <span class="hero-badge">10 features</span>
  <span class="hero-badge">4 learning profiles</span>
  <span class="hero-badge">ENSIA · ML Module 2025–2026</span>
</div>
""", unsafe_allow_html=True)

# ─── Load model ────────────────────────────────────────────────────────────────
try:
    df_data, scaler, weights, pca, kmeans = load_model(str(DATA_PATH))
    model_loaded = True
except FileNotFoundError:
    st.error(
        f"⚠️ Dataset not found at `{DATA_PATH}`. "
        "Ensure `merged_dataset.csv` is located in `Data/raw/` relative to the project root."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error while loading the model: {e}")
    st.stop()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Quick-set a profile")
    st.caption("Select a preset to load representative values, then fine-tune the sliders.")

    PRESETS = {
        "😴 Disengaged":        dict(StudyHours=16, Attendance=75, AssignmentCompletion=75,
                                     ExamScore=70, Motivation=1, StressLevel=1,
                                     Extracurricular=0, Discussions=0, OnlineCourses=10, FinalGrade=2),
        "🌟 Fully Engaged":     dict(StudyHours=35, Attendance=90, AssignmentCompletion=85,
                                     ExamScore=80, Motivation=1, StressLevel=1,
                                     Extracurricular=1, Discussions=1, OnlineCourses=10, FinalGrade=1),
        "🏃 Socially Active":   dict(StudyHours=16, Attendance=75, AssignmentCompletion=75,
                                     ExamScore=70, Motivation=1, StressLevel=1,
                                     Extracurricular=1, Discussions=0, OnlineCourses=10, FinalGrade=2),
        "📚 Academic-Focused":  dict(StudyHours=35, Attendance=90, AssignmentCompletion=85,
                                     ExamScore=80, Motivation=1, StressLevel=1,
                                     Extracurricular=0, Discussions=1, OnlineCourses=10, FinalGrade=1),
    }
    preset_choice = st.radio("", list(PRESETS.keys()), index=None,
                              label_visibility="collapsed")
    preset = PRESETS.get(preset_choice, {})

    st.divider()
    st.markdown("### 📊 Model details")
    n_clean = len(df_data)
    st.caption(
        f"**Algorithm:** K-Means (k = 4, k-means++, n_init = 30)\n\n"
        f"**Dataset:** {n_clean:,} clean records\n\n"
        f"**Features:** {len(FEATURE_COLS)} signal-aware features\n\n"
        f"**Pipeline:** MinMaxScaler → Amplification → PCA\n\n"
        f"**Silhouette Score:** ≈ 0.998"
    )

    st.divider()
    st.markdown("### 📖 How to use")
    st.caption(
        "1. Optionally select a preset profile above.\n"
        "2. Adjust the sliders and dropdowns to match your behaviour.\n"
        "3. Click **Predict my learning profile** to see your archetype."
    )

# ─── Main layout ───────────────────────────────────────────────────────────────
col_inputs, col_results = st.columns([1, 1.15], gap="large")

with col_inputs:
    st.markdown('<div class="section-hdr">📚 Academic behaviour</div>',
                unsafe_allow_html=True)

    study  = st.slider("Study hours / week",         5,  44,  preset.get("StudyHours",           20), 1)
    attend = st.slider("Attendance (%)",             60, 100, preset.get("Attendance",            80), 1)
    assign = st.slider("Assignment completion (%)",  50, 100, preset.get("AssignmentCompletion",  75), 1)
    exam   = st.slider("Exam score",                 40, 100, preset.get("ExamScore",             70), 1)
    online = st.slider("Online courses",              0,  20,  preset.get("OnlineCourses",         10), 1)

    st.markdown('<div class="section-hdr">🧠 Wellbeing & engagement</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        motiv  = st.selectbox("Motivation",    [0, 1, 2],
                               index=preset.get("Motivation",    1),
                               format_func=lambda x: ["Low", "Moderate", "High"][x])
        stress = st.selectbox("Stress level",  [0, 1, 2],
                               index=preset.get("StressLevel",   1),
                               format_func=lambda x: ["Low", "Moderate", "High"][x])
        fg     = st.selectbox("Final grade",   [0, 1, 2, 3],
                               index=preset.get("FinalGrade",    2),
                               format_func=lambda x: ["A", "B", "C", "D"][x])
    with c2:
        extra   = st.selectbox("Extracurricular", [0, 1],
                                index=preset.get("Extracurricular", 0),
                                format_func=lambda x: ["No", "Yes"][x])
        discuss = st.selectbox("Discussions",     [0, 1],
                                index=preset.get("Discussions",    0),
                                format_func=lambda x: ["No", "Yes"][x])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Predict my learning profile",
                                 use_container_width=True, type="primary")

# ─── Results panel ─────────────────────────────────────────────────────────────
with col_results:
    inputs = dict(
        StudyHours=study,    Attendance=attend,
        Extracurricular=extra, Motivation=motiv,
        OnlineCourses=online,  Discussions=discuss,
        AssignmentCompletion=assign, ExamScore=exam,
        StressLevel=stress,  FinalGrade=fg,
    )

    if predict_clicked or preset_choice:
        cid, dists = predict_cluster(inputs, scaler, weights, pca, kmeans)
        p          = PROFILES[cid]
        col_hex    = COLORS[cid]["hex"]
        col_light  = COLORS[cid]["light"]

        # ── Profile card ──────────────────────────────────────────────────────
        traits_html = "".join(
            f'<span class="trait-pill" style="background:{col_hex}22;color:{col_hex};border-color:{col_hex}44">'
            f'{t}</span>' for t in p["traits"]
        )
        st.markdown(f"""
        <div class="profile-card" style="background:{col_light};border-color:{col_hex};">
          <div class="profile-name"    style="color:{col_hex};">{p['name']}</div>
          <div class="profile-subtitle" style="color:{col_hex};">
              Cluster {cid} &nbsp;·&nbsp; {p['pct']} of students
          </div>
          <div class="profile-desc">{p['desc']}</div>
          <div style="margin-top:12px">{traits_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metric row ────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        for col_st, val, label in [
            (m1, f"{study}h",    "Study hrs/wk"),
            (m2, f"{attend}%",   "Attendance"),
            (m3, f"{assign}%",   "Assignments"),
            (m4, str(exam),      "Exam score"),
        ]:
            col_st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val" style="color:{col_hex};">{val}</div>
              <div class="metric-lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Radar chart ───────────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">📡 Your profile vs cluster average</div>',
                    unsafe_allow_html=True)

        radar_input = {k: inputs[k] for k in RADAR_LABELS}
        centroid_radar_raw = CENTROIDS_DF.loc[cid, RADAR_LABELS]

        # Build a combined frame to normalise together
        combined = pd.DataFrame(
            [radar_input, centroid_radar_raw.to_dict()],
            columns=RADAR_LABELS,
        )
        normed = pd.DataFrame(
            MinMaxScaler().fit_transform(combined),
            columns=RADAR_LABELS,
        )
        user_r     = normed.iloc[0].tolist()
        centroid_r = normed.iloc[1].tolist()

        st.plotly_chart(
            make_radar(user_r, centroid_r, cid),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── Confidence bars ───────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">📊 Match confidence — all 4 profiles</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_confidence_bar(dists),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── Personalised recommendations ──────────────────────────────────────
        st.markdown('<div class="section-hdr">💡 Personalised recommendations</div>',
                    unsafe_allow_html=True)
        for icon, text in p["recs"]:
            st.markdown(f"""
            <div class="rec-item" style="border-left-color:{col_hex};">
              <span style="font-size:1.1rem;flex-shrink:0;">{icon}</span>
              <span>{text}</span>
            </div>""", unsafe_allow_html=True)

    else:
        # ── Placeholder state ─────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    padding:5rem 1rem;opacity:0.4;text-align:center;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">🎓</div>
          <div style="font-size:1rem;color:#64748B;max-width:280px;line-height:1.65;">
            Adjust the sliders on the left, then click<br>
            <strong>Predict my learning profile</strong> to see your results.
          </div>
        </div>""", unsafe_allow_html=True)

# ─── Profile overview section ──────────────────────────────────────────────────
st.divider()
st.markdown("### 🗂️ All four learning profiles at a glance")

ov_cols = st.columns(4)
for i, (cid, p) in enumerate(PROFILES.items()):
    col_hex   = COLORS[cid]["hex"]
    col_light = COLORS[cid]["light"]
    signals   = " &nbsp;·&nbsp; ".join(p["key_signals"])
    with ov_cols[i]:
        st.markdown(f"""
        <div class="overview-card"
             style="background:{col_light};
                    border-top-color:{col_hex};
                    border-left-color:{col_hex}33;
                    border-right-color:{col_hex}33;
                    border-bottom-color:{col_hex}33;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;
                      color:{col_hex};margin-bottom:2px;">{p['name']}</div>
          <div style="font-size:0.75rem;color:{col_hex};opacity:.75;margin-bottom:8px;">
              {p['pct']} of students</div>
          <div style="font-size:0.82rem;color:#334155;line-height:1.55;margin-bottom:8px;">
              {p['desc'][:115]}…</div>
          <div style="font-size:0.72rem;color:{col_hex};opacity:.85;font-style:italic;">
              {signals}</div>
        </div>""", unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 0.5rem;
            font-size:0.78rem;color:#94A3B8;border-top:1px solid #E2E8F0;margin-top:1rem;">
  Student Learning Profile Predictor &nbsp;·&nbsp; ENSIA ML Module 2025–2026
  &nbsp;·&nbsp; K-Means · k = 4 · 12,469 students
</div>
""", unsafe_allow_html=True)
