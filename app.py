import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(page_title="SkillDelta", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f0f0f; color: #f0f0f0; }
    .block-container { padding: 2rem 3rem; }
    h1, h2, h3 { color: #f0f0f0; }
    .stTextInput > div > div > input { background-color: #1a1a1a; color: #f0f0f0; border: 1px solid #333; }
    .stTextArea > div > div > textarea { background-color: #1a1a1a; color: #f0f0f0; border: 1px solid #333; }
    .stSelectbox > div > div { background-color: #1a1a1a; color: #f0f0f0; }
    div[data-testid="metric-container"] { background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jobs(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    df['skills'] = df['skills'].str.strip()
    return df

def parse_skills(raw):
    return [s.strip().lower() for s in raw.replace(',', ' ').split() if s.strip()]

def match_score(student_skills, job_skills_str):
    docs = [' '.join(student_skills), job_skills_str.lower()]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(docs)
    score = cosine_similarity(matrix[0], matrix[1])[0][0]
    return round(score * 100, 1)

def get_missing(student_skills, job_skills_str):
    job_set = set(job_skills_str.lower().split())
    student_set = set(student_skills)
    return sorted(job_set - student_set)

# ── sidebar: student profile ──────────────────────────────────────────────────

with st.sidebar:
    st.title("SkillDelta")
    st.markdown("---")
    st.subheader("Student Profile")
    name = st.text_input("Name")
    branch = st.text_input("Branch")
    skills_raw = st.text_area("Your Skills (comma or space separated)", placeholder="Python SQL HTML Java")
    target_role = st.text_input("Target Job Role", placeholder="Data Analyst")
    run = st.button("Analyze", use_container_width=True)

# ── main area ─────────────────────────────────────────────────────────────────

st.title("Skill Gap Analysis Dashboard")
st.markdown("---")

uploaded = st.file_uploader("Upload Job Dataset CSV", type=["csv"])

if uploaded:
    jobs_df = load_jobs(uploaded)
    st.success(f"{len(jobs_df)} jobs loaded")

    if run and skills_raw:
        student_skills = parse_skills(skills_raw)

        # score every job
        jobs_df['score'] = jobs_df['skills'].apply(lambda s: match_score(student_skills, s))
        jobs_df['missing'] = jobs_df['skills'].apply(lambda s: get_missing(student_skills, s))
        jobs_df_sorted = jobs_df.sort_values('score', ascending=False).reset_index(drop=True)

        # target role analysis
        target_row = jobs_df[jobs_df['job_title'].str.lower() == target_role.strip().lower()]

        st.markdown("---")
        st.subheader(f"Results for {name}" if name else "Results")

        # ── row 1: metrics ────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Skills", len(student_skills))
        with col2:
            if not target_row.empty:
                st.metric("Match Score — " + target_role, f"{target_row.iloc[0]['score']}%")
            else:
                st.metric("Match Score", "Role not found")
        with col3:
            if not target_row.empty:
                missing = target_row.iloc[0]['missing']
                st.metric("Missing Skills", len(missing))

        st.markdown("---")

        # ── row 2: gap analysis + recommendations ─────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Skill Gap — " + (target_role if target_role else "Target Role"))
            if not target_row.empty:
                job_skills_list = target_row.iloc[0]['skills'].lower().split()
                matched = [s for s in job_skills_list if s in student_skills]
                missing = target_row.iloc[0]['missing']

                st.markdown("**Matched Skills**")
                for s in matched:
                    st.markdown(f"- {s.title()}")

                st.markdown("**Missing Skills**")
                for s in missing:
                    st.markdown(f"- {s.title()}")
            else:
                st.info("Target role not found in dataset. Check spelling.")

        with col_right:
            st.subheader("Top Job Matches")
            for _, row in jobs_df_sorted.head(5).iterrows():
                st.markdown(f"**{row['job_title']}** — {row['score']}%")
                st.progress(int(row['score']) / 100)

        st.markdown("---")

        # ── row 3: skill recommendations ─────────────────────────────────────
        st.subheader("Skills to Learn Next")
        all_missing = []
        for m in jobs_df_sorted.head(5)['missing']:
            all_missing.extend(m)
        freq = Counter(all_missing).most_common(8)

        if freq:
            rec_col1, rec_col2 = st.columns(2)
            for i, (skill, count) in enumerate(freq):
                if i % 2 == 0:
                    rec_col1.markdown(f"- {skill.title()} — appears in {count} top roles")
                else:
                    rec_col2.markdown(f"- {skill.title()} — appears in {count} top roles")
        else:
            st.success("No major skill gaps in your top matches.")

        st.markdown("---")

        # ── row 4: charts ─────────────────────────────────────────────────────
        st.subheader("Analytics")

        chart1, chart2 = st.columns(2)

        with chart1:
            all_skills = ' '.join(jobs_df['skills']).lower().split()
            skill_freq = Counter(all_skills).most_common(10)
            skill_df = pd.DataFrame(skill_freq, columns=['Skill', 'Count'])
            fig1 = px.bar(skill_df, x='Count', y='Skill', orientation='h',
                          title='Most In-Demand Skills',
                          color='Count', color_continuous_scale='Blues',
                          template='plotly_dark')
            st.plotly_chart(fig1, use_container_width=True)

        with chart2:
            fig2 = px.bar(jobs_df_sorted.head(8), x='score', y='job_title',
                          orientation='h', title='Your Job Match Scores (%)',
                          color='score', color_continuous_scale='Teal',
                          template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)

    elif run and not skills_raw:
        st.warning("Enter your skills in the sidebar first.")

else:
    st.info("Upload a jobs CSV file above to get started.")