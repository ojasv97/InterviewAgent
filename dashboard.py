import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = "jd.db"  # adjust if different


def get_score_summary():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT c.id as candidate_id, c.name, jd.id as jd_id, jd.jd_text,
               s.overall_score
        FROM candidate_score_summary s
        JOIN candidate c ON s.candidate_id = c.id
        JOIN job_description jd ON s.jd_id = jd.id
    """, conn)
    conn.close()
    return df


def get_score_detail(candidate_id, jd_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT q.interview_question_text, d.keyword_overlap_pct,
               d.semantic_similarity, d.llm_score, d.final_score
        FROM candidate_score_detail d
        JOIN interview_question q ON d.question_id = q.id
        WHERE d.candidate_id=? AND d.jd_id=?
    """, conn, params=(candidate_id, jd_id))
    conn.close()
    return df


# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="HR Candidate Scoring", layout="wide")

st.title("📊 Candidate Scoring Dashboard")

summary_df = get_score_summary()

if summary_df.empty:
    st.warning("No scoring data available yet.")
else:
    # Show summary table
    st.subheader("Overall Candidate Scores")
    st.dataframe(summary_df, use_container_width=True)

    # Candidate selector
    st.subheader("🔍 Drilldown into Candidate Responses")
    selected_row = st.selectbox(
        "Select Candidate", 
        summary_df.apply(lambda x: f"{x['name']} (JD {x['jd_id']}, Score={x['overall_score']})", axis=1)
    )

    if selected_row:
        # Extract candidate_id & jd_id from selection
        idx = summary_df.apply(lambda x: f"{x['name']} (JD {x['jd_id']}, Score={x['overall_score']})", axis=1).tolist().index(selected_row)
        candidate_id = int(summary_df.iloc[idx]["candidate_id"])
        jd_id = int(summary_df.iloc[idx]["jd_id"])

        details_df = get_score_detail(candidate_id, jd_id)

        if not details_df.empty:
            st.write(f"### Detailed Scores for {summary_df.iloc[idx]['name']} (JD {jd_id})")
            st.dataframe(details_df, use_container_width=True)

            # Show charts
            st.bar_chart(details_df.set_index("interview_question_text")[["final_score"]])
        else:
            st.info("No detailed question scores available for this candidate.")