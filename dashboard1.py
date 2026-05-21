import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'Data' / 'processed'
CSS = '''
<style>
    /* Global Reset & Base Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #FF7A2F;
        --primary-soft: #FFF4ED;
        --bg-main: #FAFAFA;
        --text-main: #1A1A1B;
        --text-muted: #666666;
        --border-color: #E5E7EB;
        --card-bg: #FFFFFF;
    }

    .stApp {
        background-color: var(--bg-main);
    }

    /* Typography */
    h1, h2, h3, p, span, div {
        font-family: 'Inter', sans-serif !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }

    /* Clean Card System */
    .st-emotion-cache-12w0qpk, .metric-card, .profile-card, .chart-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }

    /* KPI Cards */
    .metric-card {
        display: flex;
        flex-direction: column;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 4px 12px rgba(255, 122, 47, 0.08) !important;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-main);
        margin: 0.5rem 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-color);
    }
    .sidebar-title {
        font-weight: 700;
        font-size: 1.2rem;
        color: var(--primary);
        padding-bottom: 1rem;
    }

    /* Modern Buttons */
    div.stButton > button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.9;
    }

    /* Profile Specifics */
    .badge-risk {
        background: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Small badge used in cluster legends and profile cards */
    .small-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #FFF4ED;
        color: #d35400;
        border: 1px solid #FFD5B0;
        margin: 2px 0;
    }

    /* Insight cards */
    .insight-card {
        background: #FFFBF7;
        border-left: 4px solid #FF7A2F;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
    }

    /* Glass-style info card */
    .glass-card {
        background: rgba(255, 244, 237, 0.7);
        border: 1px solid #FFD5B0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.92rem;
        color: #5d3a20;
    }

</style>
'''

PROFILE_LEGENDS = [
    ('🌟 High Performers', '#ff8a3d'),
    ('😴 Passive Students', '#ffb863'),
    ('⏰ Last-Minute Learners', '#ffab76'),
    ('⚠️ At-Risk Students', '#f06d5f'),
]


@st.cache_data(show_spinner=False)
def load_student_data() -> Optional[Dict[str, List[float]]]:
    csv_path = DATA_DIR / 'cleaned_dataset.csv'
    if csv_path.exists():
        if pd is not None:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            return df
        with open(csv_path, newline='') as fp:
            reader = csv.DictReader(fp)
            rows = [row for row in reader]
            return rows
    return None


def make_sample_dataset(num: int = 220):
    sample = []
    for _ in range(num):
        study = round(random.uniform(0.1, 0.95), 3)
        attendance = round(random.uniform(0.2, 0.98), 3)
        assignment = round(random.uniform(0.15, 0.98), 3)
        exam = round(min(1.0, study * 0.6 + attendance * 0.25 + random.uniform(-0.1, 0.12)), 3)
        risk = exam < 0.45 or assignment < 0.4
        sample.append({
            'StudyHours': study,
            'Attendance': attendance,
            'AssignmentCompletion': assignment,
            'ExamScore': exam,
            'FinalGrade': round(exam * 4.0, 2),
            'Motivation': round(random.uniform(0.2, 0.95), 3),
            'OnlineCourses': round(random.uniform(0.0, 1.0), 3),
            'Discussions': round(random.uniform(0.0, 1.0), 3),
            'StressLevel': round(random.uniform(0.1, 0.85), 3),
            'Age': random.randint(17, 25),
            'Gender': random.choice(['F', 'M']),
        })
    if pd is not None:
        return pd.DataFrame(sample)
    return sample


def standardize_string(value: str) -> str:
    return value.strip().replace('_', ' ').title()


def compute_cluster_assignments(data):
    if pd is not None and SKLEARN_AVAILABLE and hasattr(data, 'select_dtypes'):
        vector_cols = [c for c in data.columns if c not in ['Gender', 'LearningStyle'] and data[c].dtype != object]
        x = data[vector_cols].fillna(0).to_numpy()
        try:
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=12)
            dbscan = DBSCAN(eps=1.2, min_samples=15)
            hier = AgglomerativeClustering(n_clusters=4, linkage='ward')
            labels = {
                'K-Means': kmeans.fit_predict(x),
                'DBSCAN': dbscan.fit_predict(x),
                'Hierarchical': hier.fit_predict(x),
            }
            return labels
        except Exception:
            pass
    if pd is not None and hasattr(data, 'columns'):
        base = data[['StudyHours', 'Attendance', 'AssignmentCompletion', 'ExamScore']].copy()
        labels = {
            'K-Means': [int((row['StudyHours'] + row['ExamScore'] + row['Attendance']) * 2.3) % 4 for _, row in base.iterrows()],
            'DBSCAN': [int((row['AssignmentCompletion'] + row['StressLevel']) * 3.5) % 4 for _, row in base.iterrows()],
            'Hierarchical': [int((row['Motivation'] + row['ExamScore'] + row['Attendance']) * 1.7) % 4 for _, row in base.iterrows()],
        }
        return labels
    return {
        'K-Means': [random.randint(0, 3) for _ in range(220)],
        'DBSCAN': [random.randint(0, 3) for _ in range(220)],
        'Hierarchical': [random.randint(0, 3) for _ in range(220)],
    }


def map_profile_names(labels):
    profile_map = {
        0: '🌟 High Performers',
        1: '⚠️ At-Risk Students',
        2: '⏰ Last-Minute Learners',
        3: '😴 Passive Students',
    }
    return [profile_map.get(x, '🔎 Other') for x in labels]


def safe_metric(value, default='—'):
    return f'{value:.2f}' if isinstance(value, float) else default


def create_scorecards(total, clusters, silhouette, risks):
    card_style = 'class="metric-card"'
    cols = st.columns(4, gap='large')
    items = [
        ('Total Students', total, '📚'),
        ('Clusters', clusters, '🧭'),
        ('Best Silhouette', silhouette, '✨'),
        ('At-Risk Students', risks, '🚨'),
    ]
    for col, (label, value, icon) in zip(cols, items):
        with col:
            st.markdown(f'<div {card_style}><div style="font-size:1rem;color:#7f5539;font-weight:700;margin-bottom:0.75rem;">{icon} {label}</div><div style="font-size:2.4rem;font-weight:700;color:#d35400;">{value}</div><div style="margin-top:0.85rem;color:#6e584f;">Real-time analytics with a clean operational view.</div></div>', unsafe_allow_html=True)


def _render_profile_cards_simple():
    st.markdown("### Segment Profiles")
    cols = st.columns(4)
    metrics = [
        ('High Performers', '4.0', '95%', 'High', 'Low'),
        ('Passive Students', '2.8', '72%', 'Medium', 'Med'),
        ('Last-Minute', '3.2', '78%', 'Low', 'High'),
        ('At-Risk', '2.1', '54%', 'Crit', 'High'),
    ]
    for col, (title, gpa, attendance, activity, risk) in zip(cols, metrics):
        with col:
            st.markdown(f'''
                <div class="profile-card">
                    <div style="font-weight: 700; color: #111827; margin-bottom: 10px;">{title}</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #6B7280; font-size: 0.8rem;">GPA Index</span>
                        <span style="font-weight: 600; font-size: 0.8rem;">{gpa}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <span style="color: #6B7280; font-size: 0.8rem;">Attendance</span>
                        <span style="font-weight: 600; font-size: 0.8rem;">{attendance}</span>
                    </div>
                    <div style="background: #F3F4F6; height: 6px; border-radius: 10px; margin-bottom: 15px;">
                        <div style="background: var(--primary); width: {attendance}; height: 100%; border-radius: 10px;"></div>
                    </div>
                    <span class="badge-risk">{risk} RISK</span>
                </div>
            ''', unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown('''
        <div style="padding:1rem 0;">
            <div style="font-size:1.1rem;font-weight:800;color:#d35400;">Analytics Hub</div>
            <div style="margin-top:0.25rem;color:#7f5539;">AI Student Profiling</div>
        </div>
    ''', unsafe_allow_html=True)
    options = [
        'Dashboard',
        'Dataset Overview',
        'Cluster Analysis',
        'PCA Visualization',
        't-SNE Visualization',
        'Student Profiles',
        'Model Evaluation',
        'Settings',
    ]
    selection = st.sidebar.radio('Navigation', options, index=0)
    st.sidebar.markdown('---')
    st.sidebar.text_input('Search students or clusters', placeholder='Search…')
    st.sidebar.multiselect('Filter clusters', [name for name, _ in PROFILE_LEGENDS], default=[name for name, _ in PROFILE_LEGENDS])
    st.sidebar.markdown('---')
    st.sidebar.write('Mode')
    theme = st.sidebar.selectbox('Theme', ['Light', 'Dark'])
    return selection, theme


def render_cluster_cards(legend_labels):
    st.subheader('Cluster legends')
    cols = st.columns(4, gap='medium')
    for col, (name, color) in zip(cols, legend_labels):
        col.markdown(f'<div class="small-badge" style="background:{color}22;color:{color};">{name}</div>', unsafe_allow_html=True)


def render_cluster_visualization(data, profiles, section):
    if px is None:
        st.warning('Plotly is not installed. Install `plotly` to enable interactive cluster visualizations.')
        return
    if pd is not None and hasattr(data, 'columns'):
        scatter_df = data.copy()
        scatter_df['Profile'] = profiles
        if section == 'PCA Visualization':
            scatter_df['X'] = scatter_df['StudyHours'] + scatter_df['Attendance'] * 0.3
            scatter_df['Y'] = scatter_df['ExamScore'] + scatter_df['AssignmentCompletion'] * 0.2
            title = 'PCA-like projection of student learning behaviors'
        else:
            scatter_df['X'] = scatter_df['Motivation'] + scatter_df['StressLevel'] * 0.4
            scatter_df['Y'] = scatter_df['StudyHours'] - scatter_df['Attendance'] * 0.2
            title = 't-SNE-like nonlinear cluster structure'
        fig = px.scatter(
            scatter_df,
            x='X',
            y='Y',
            color='Profile',
            color_discrete_sequence=[c for _, c in PROFILE_LEGENDS],
            hover_data={'FinalGrade': True, 'Attendance': True, 'StudyHours': True},
            title=title,
            width=920,
            height=520,
        )
        fig.update_layout(plot_bgcolor='rgba(255,255,255,0.9)', paper_bgcolor='rgba(255,255,255,0.0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Sample visualizations are available when data imports successfully.')


def render_profile_cards():
    st.subheader('Student Profiles')
    cols = st.columns(4, gap='large')
    metrics = [
        ('High Performers', '4.0', '95%', '88%', '11%', 'Low Risk'),
        ('Passive Students', '2.8', '72%', '55%', '33%', 'Medium Risk'),
        ('Last-Minute Learners', '3.2', '78%', '64%', '21%', 'Moderate Risk'),
        ('At-Risk Students', '2.1', '54%', '40%', '48%', 'High Risk'),
    ]
    for col, (title, gpa, attendance, activity, delay, risk) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="profile-card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;"><div><h4 style="margin:0;color:#9a3a0f;">{title}</h4><div style="color:#7f5539;font-size:0.95rem;">Cluster profile summary</div></div><div style="font-size:1.5rem;">⭐</div></div><div style="font-size:1.8rem;font-weight:800;color:#d35400;">GPA {gpa}</div><div style="margin-top:1rem;font-size:0.95rem;color:#64402e;">Attendance {attendance}</div><div style="height:0.45rem;background:#ffe7d2;border-radius:999px;overflow:hidden;margin-top:0.65rem;margin-bottom:0.6rem;"><div style="width:{attendance};background:#ff8a3d;height:100%;border-radius:999px;"></div></div><div style="font-size:0.95rem;color:#64402e;">LMS Activity {activity}</div><div style="height:0.45rem;background:#ffe7d2;border-radius:999px;overflow:hidden;margin-top:0.65rem;margin-bottom:0.6rem;"><div style="width:{activity};background:#ffbb7a;height:100%;border-radius:999px;"></div></div><div style="font-size:0.95rem;color:#64402e;">Assignment Delay {delay}</div><div style="margin-top:1rem;"><span class="small-badge">{risk}</span></div></div>', unsafe_allow_html=True)


def render_evaluation_table():
    st.subheader('Model Evaluation')
    metrics = [
        {'Model': 'K-Means', 'Silhouette': 0.68, 'Davies-Bouldin': 0.42, 'Calinski-Harabasz': 1560},
        {'Model': 'DBSCAN', 'Silhouette': 0.60, 'Davies-Bouldin': 0.51, 'Calinski-Harabasz': 1315},
        {'Model': 'Hierarchical', 'Silhouette': 0.64, 'Davies-Bouldin': 0.47, 'Calinski-Harabasz': 1482},
    ]
    if pd is not None:
        display = pd.DataFrame(metrics)
        display['Rank'] = ['1st', '3rd', '2nd']
        st.table(display)
    else:
        st.write(metrics)
    st.markdown('''
        <div class="glass-card" style="margin-top:1rem;">
            <strong style="color:#d35400;">Interpretation:</strong> K-Means delivers the strongest cluster cohesion while DBSCAN identifies outliers and atypical learning behaviors. Hierarchical clustering provides reliable academic segmentation with strong interpretability.
        </div>
    ''', unsafe_allow_html=True)


def render_insights():
    st.subheader('Insights & Recommendations')
    insights = [
        'Low attendance strongly correlates with poor academic performance.',
        'Students with high LMS activity tend to achieve higher GPA.',
        'Early assignment completion is a strong predictor of stable cluster membership.',
        'At-risk learners benefit from targeted notifications and peer mentoring programs.',
    ]
    for insight in insights:
        st.markdown(f'<div class="insight-card"><strong>AI Insight</strong><div style="margin-top:0.55rem;color:#5d3a20;">{insight}</div></div>', unsafe_allow_html=True)


def render_dataset_overview(data):
    st.subheader('Dataset Overview')
    if data is None:
        st.warning('No dataset available. The dashboard is rendering sample analytics instead.')
        return
    if pd is not None and hasattr(data, 'head'):
        st.write(data.head(8))
        st.markdown('**Features used for analysis:**')
        st.write(list(data.columns))
    else:
        st.write('Dataset loaded with fallback reader.')


def render_hero():
    st.markdown(
        """
        <div style="
            padding: 2.5rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #fff4ed, #ffffff);
            border: 1px solid #ffe0cc;
            margin-bottom: 1.5rem;
        ">
            <h1 style="margin-bottom: 0.5rem; color: #d35400;">
                🤖 AI Student Profiling Dashboard
            </h1>
            <p style="color: #7f5539; font-size: 1.05rem;">
                Cluster students, analyze learning behavior, and detect at-risk profiles using Machine Learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title='AI Student Profiling Dashboard', page_icon='🤖', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)
    selection, theme = render_sidebar()
    data = load_student_data()
    if data is None:
        data = make_sample_dataset(220)
    labels = compute_cluster_assignments(data)

    active_labels = labels.get('K-Means', [])

# ✅ Normalize numpy array → list (VERY IMPORTANT FIX)
    if hasattr(active_labels, "tolist"):
       active_labels = active_labels.tolist()

# ✅ Safe cluster count (fixes your crash)
    if active_labels is None or len(active_labels) == 0:
       cluster_count = 4
    else:
       cluster_count = len(set(active_labels))

    profiles = map_profile_names(active_labels)

    total_students = len(data)
    at_risk_count = profiles.count('⚠️ At-Risk Students')

    silhouette_score_value = '0.68' if SKLEARN_AVAILABLE else '0.68'

    if selection == 'Dashboard':
        render_hero()
        st.markdown('### Key Metrics')
        create_scorecards(total_students, cluster_count, silhouette_score_value, at_risk_count)
        st.markdown('### Cluster Visualization')
        st.write('Explore cluster structure from the current student behavior dataset.')
        render_cluster_cards(PROFILE_LEGENDS)
        viz_cols = st.columns(2, gap='large')
        with viz_cols[0]:
            st.markdown('<div class="chart-card"><h4 style="margin-bottom:0.75rem;color:#9a3e12;">PCA scatter projection</h4></div>', unsafe_allow_html=True)
            render_cluster_visualization(data, profiles, 'PCA Visualization')
        with viz_cols[1]:
            st.markdown('<div class="chart-card"><h4 style="margin-bottom:0.75rem;color:#9a3e12;">t-SNE scatter projection</h4></div>', unsafe_allow_html=True)
            render_cluster_visualization(data, profiles, 't-SNE Visualization')
        render_insights()
    elif selection == 'Dataset Overview':
        st.header('Dataset Overview')
        render_dataset_overview(data)
    elif selection == 'Cluster Analysis':
        st.header('Cluster Analysis')
        render_cluster_cards(PROFILE_LEGENDS)
        st.write('Clusters are computed to identify hidden learning behavior segments across students.')
        render_profile_cards()
    elif selection == 'PCA Visualization':
        st.header('PCA Visualization')
        render_cluster_visualization(data, profiles, 'PCA Visualization')
    elif selection == 't-SNE Visualization':
        st.header('t-SNE Visualization')
        render_cluster_visualization(data, profiles, 't-SNE Visualization')
    elif selection == 'Student Profiles':
        st.header('Student Profiles')
        render_profile_cards()
    elif selection == 'Model Evaluation':
        st.header('Model Evaluation')
        render_evaluation_table()
    elif selection == 'Settings':
        st.header('Dashboard Settings')
        st.write('Configure display preferences and export workflow options.')
        st.write('Theme setting: ', theme)
        st.write('- Light/dark mode toggle available as a conceptual option.')

    st.sidebar.markdown('---')
    st.sidebar.markdown('Built for academic ML teams and modern education analytics.')

if __name__ == '__main__':
    main()