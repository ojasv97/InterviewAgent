"""
interview_scoring.py

Implements scoring pipeline:

40% keyword/concept overlap  
30% semantic similarity (embeddings)  
30% LLM qualitative score (GPT-4)

Stores interview results into SQLite (jd.db) with tables:
- candidate_score_detail
- candidate_score_summary

Usage:
Ensure all Azure OpenAI environment variables are set in .env
Put candidates in candidate table with is_scheduled='yes'
Run: python interview_scoring.py
"""

import os
import re
import sqlite3
import json
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np
from openai import AzureOpenAI

# --------------------------
# Environment / LLM client
# --------------------------

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT")
AZURE_DEPLOYMENT_EMBED = os.getenv("AZURE_DEPLOYMENT_EMBED")

if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_DEPLOYMENT_CHAT and AZURE_DEPLOYMENT_EMBED):
    raise RuntimeError("Missing one or more Azure OpenAI env vars in .env")

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
    # key picked up from AZURE_OPENAI_API_KEY
)

DB_PATH = "jd.db"

# --------------------------
# DB helpers
# --------------------------

def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Detail table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidate_score_detail (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER NOT NULL,
        jd_id INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        keyword_overlap_pct REAL,
        semantic_similarity REAL,
        llm_score REAL,
        final_score REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (candidate_id) REFERENCES candidate(id) ON DELETE CASCADE,
        FOREIGN KEY (jd_id) REFERENCES job_description(id) ON DELETE CASCADE,
        FOREIGN KEY (question_id) REFERENCES interview_question(id) ON DELETE CASCADE
    )
    """)

    # Summary table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidate_score_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER NOT NULL,
        jd_id INTEGER NOT NULL,
        overall_score REAL NOT NULL,
        is_best BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (candidate_id) REFERENCES candidate(id) ON DELETE CASCADE,
        FOREIGN KEY (jd_id) REFERENCES job_description(id) ON DELETE CASCADE
    )
    """)
    conn.commit()
    conn.close()

# --------------------------
# Skills extraction
# --------------------------

GENERIC_SKILLS = [
    # Core
    "python", "sql", "spark", "pyspark", "hadoop", "airflow", "etl", "data engineering",
    "data warehouse", "snowflake", "redshift", "bigquery", "dbt", "tableau", "power bi",
    "aws", "azure", "gcp", "databricks", "machine learning", "ml", "statistics",
    "testing", "qa", "automation", "ci/cd", "kafka", "nosql", "mongodb", "api", "rest", "s3",
    # Pharma / analytics
    "forecasting", "call planning", "statistical modelling", "predictive analytics",
    "real world evidence", "rwe", "patient data", "hcp", "hco", "master data management",
    "mdm", "veeva", "iqvia", "sas", "pharmacovigilance", "market mix modelling"
]

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s+/#.-]', ' ', (s or "").lower())).strip()

def extract_skills_from_text(text: str, llm_client=None) -> List[str]:
    if not text:
        return []

    text_norm = normalize_text(text)
    found = set()

    # dictionary-based
    for skill in GENERIC_SKILLS:
        if skill.lower() in text_norm:
            found.add(skill.lower())

    # LLM semantic augmentation
    if llm_client:
        try:
            prompt = f"""
            Extract important skills, tools, and pharma domain terms from this job description.
            Return a comma-separated list only.
            JD:
            {text}
            """
            resp = llm_client.chat.completions.create(
                model=AZURE_DEPLOYMENT_CHAT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            llm_skills = resp.choices[0].message.content.strip().lower()
            for s in re.split(r'[,;\n]', llm_skills):
                s = s.strip()
                if s:
                    found.add(s)
        except Exception as e:
            print("⚠️ LLM skill extraction failed:", e)

    return sorted(found)

def merge_jd_skills(jd_text: str, manual_skills: str, llm_client=None) -> List[str]:
    skills = set()
    if manual_skills and manual_skills.strip():
        for s in manual_skills.split(","):
            if s.strip():
                skills.add(s.strip().lower())
    extracted = extract_skills_from_text(jd_text, llm_client=llm_client)
    skills.update(extracted)
    return sorted(skills)

# --------------------------
# Scoring helpers
# --------------------------

def keyword_overlap_score(jd_skills: List[str], ans_skills: List[str]) -> float:
    jd_set = set(s.lower() for s in jd_skills if s)
    ans_set = set(s.lower() for s in ans_skills if s)
    if not jd_set:
        return 0.0
    return len(jd_set & ans_set) / len(jd_set)

def get_embedding(text: str) -> List[float]:
    try:
        resp = client.embeddings.create(model=AZURE_DEPLOYMENT_EMBED, input=text)
        return resp.data[0].embedding
    except Exception as e:
        print("⚠️ Embedding error:", e)
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / denom) if denom else 0.0

def semantic_similarity_score(jd_text: str, ans_text: str) -> float:
    if not jd_text.strip() or not ans_text.strip() or len(ans_text.split()) < 5:
        return 0.0
    emb_jd = get_embedding(jd_text)
    emb_ans = get_embedding(ans_text)
    sim = cosine_similarity(emb_jd, emb_ans)
    return round((sim + 1) / 2 * 100, 2)

def llm_evaluate_answer(jd_text: str, question: str, ans_text: str) -> Tuple[float, str]:
    system = "You are an expert technical interviewer and evaluator. Be concise and only return structured output."
    user_prompt = (
        "Job Description:\n"
        f"{jd_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{ans_text}\n\n"
        "Please do two things:\n"
        "1) Provide a one-line 'internet-style' example answer (a concise ideal answer line).\n"
        "2) Provide a numeric score 0-100 evaluating how well the candidate answer matches the job description and "
        "how plausible/credible it is (consider technical depth, specificity, evidence of ownership, tools). "
        "Return output as JSON exactly with keys: internet_answer, llm_score\n"
        "Example output:\n"
        '{"internet_answer": "One-liner ideal answer...", "llm_score": 78}\n'
        "Only return the JSON object, nothing else."
    )
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_CHAT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
            max_tokens=150
        )
        text = resp.choices[0].message.content.strip()
        parsed = json.loads(re.search(r"\{.*\}", text, flags=re.DOTALL).group(0))
        return float(parsed.get("llm_score", 0)), parsed.get("internet_answer", "")
    except Exception as e:
        print("⚠️ LLM eval failed:", e)
        return 0.0, ""

def sensible_and_doable_flags(answer_text: str) -> Tuple[str, str]:
    sensible_verbs = [
        "implemented", "deployed", "designed", "led", "managed", "built", "developed",
        "created", "automated", "scaled", "optimized", "engineered", "analyzed",
        "architected", "improved", "delivered", "executed", "launched", "configured",
        "maintained", "tested", "resolved", "enhanced", "streamlined", "orchestrated",
        "validated", "monitored", "integrated", "customized", "coordinated",
        "documented", "evaluated", "collaborated", "transformed", "migrated",
        "standardized", "secured", "modernized", "debugged", "facilitated",
        "trained", "supported", "implemented compliance", "ensured", "oversaw"
    ]
    role_keywords = [
        "python", "sql", "machine learning", "statistics", "pipeline",
        "model", "dashboard", "kpi", "reporting", "data", "etl",
        "big data", "spark", "databricks", "hadoop", "airflow",
        "snowflake", "redshift", "data warehouse", "data lake",
        "power bi", "tableau", "visualization", "analytics",
        "regression", "classification", "forecasting", "optimization",
        "api", "integration", "cloud", "aws", "azure", "gcp",
        "scalability", "automation", "orchestration", "monitoring",
        "clinical trials", "patient data", "sales data", "hcp", "hco",
        "biostatistics", "compliance", "gxp", "data governance",
        "feature engineering", "nlp", "time series", "data quality",
        "normalization", "transformation", "joins", "query optimization",
        "security", "encryption", "metadata", "version control", "git"
    ]
    ans = (answer_text or "").strip().lower()
    def is_sensible(ans: str) -> str:
        words = ans.split()
        if len(words) < 5:
            return "No"
        if not any(k in ans for k in role_keywords):
            return "No"
        if not any(v in ans for v in sensible_verbs):
            return "No"
        if ans in ["i did some work", "i worked", "i helped"]:
            return "No"
        return "Yes"
    sensible = is_sensible(ans)
    doable_verbs = [
        "executed", "launched", "orchestrated", "delivered", "completed", "streamlined",
        "finalized", "piloted", "operationalized", "engineered", "constructed", "formulated",
        "architected", "crafted", "initiated", "innovated", "established", "introduced",
        "optimized", "enhanced", "upgraded", "modernized", "refined", "accelerated",
        "boosted", "strengthened", "evolved", "directed", "coordinated", "facilitated",
        "mentored", "supervised", "guided", "collaborated", "partnered", "delegated",
        "analyzed", "evaluated", "investigated", "assessed", "measured", "benchmarked",
        "diagnosed", "forecasted", "quantified", "resolved", "transformed", "improved",
        "consolidated", "restructured", "migrated", "transitioned", "mitigated", "customized",
        "implemented", "deployed", "designed", "led", "managed", "built",
        "developed", "created", "automated", "scaled"
    ]
    doable = "Yes" if any(v in ans for v in doable_verbs) else "No"
    return sensible, doable

def aggregate_final_score(keyword_frac: float, semantic_sim: float, llm_score: float, doable: str, sensible: str) -> float:
    # Scale weights properly
    k = 40 * keyword_frac          # keyword_frac is a ratio (0–1), so scale to 0–40
    s = 30 * (semantic_sim / 100)  # semantic_sim already 0–100, normalize then weight
    l = 30 * (llm_score / 100)     # llm_score already 0–100, normalize then weight
    
    base = k + s + l

    # Add a small bonus (5%) if answer is sensible/doable
    if sensible == "Yes" or doable == "Yes":
        final = base * 1.05
    else:
        final = base

    return round(final, 2)
# --------------------------
# Scoring flow
# --------------------------
import sqlite3

def update_best_candidates(db_path: str = DB_PATH):
    """
    For each jd_id, set is_best=True for the candidate(s) with the highest overall_score.
    If only one candidate exists for a jd_id, mark them best by default.
    Uses UPSERT-like logic to ensure consistency.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get all jd_ids
    cur.execute("SELECT DISTINCT jd_id FROM candidate_score_summary")
    jd_ids = [row[0] for row in cur.fetchall()]

    for jd_id in jd_ids:
        # Get candidate scores for this jd_id
        cur.execute("""
            SELECT candidate_id, overall_score 
            FROM candidate_score_summary 
            WHERE jd_id=?
        """, (jd_id,))
        candidates = cur.fetchall()

        if not candidates:
            continue

        if len(candidates) == 1:
            # Only one candidate → mark as best
            candidate_id, _ = candidates[0]
            cur.execute("""
                UPDATE candidate_score_summary
                SET is_best=1
                WHERE jd_id=? AND candidate_id=?
            """, (jd_id, candidate_id))
        else:
            # Find max score
            max_score = max(score for _, score in candidates)

            # Reset all to false
            cur.execute("UPDATE candidate_score_summary SET is_best=0 WHERE jd_id=?", (jd_id,))

            # Mark only highest scoring candidates as best
            cur.execute("""
                UPDATE candidate_score_summary
                SET is_best=1
                WHERE jd_id=? AND overall_score=?
            """, (jd_id, max_score))

    conn.commit()
    conn.close()


def calculate_score(candidate_id: int, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT c.name, c.jd_id, jd.jd_text, jd.skills
        FROM candidate c
        JOIN job_description jd ON c.jd_id = jd.id
        WHERE c.id=?
    """, (candidate_id,))
    row = cur.fetchone()
    if not row:
        print(f"No candidate {candidate_id}")
        return
    name, jd_id, jd_text, manual_skills = row
    print(f"\nInterviewing {name} for JD {jd_id}")

    jd_skills = merge_jd_skills(jd_text, manual_skills, llm_client=client)

    cur.execute("SELECT id, interview_question_text FROM interview_question WHERE jd_id=?", (jd_id,))
    questions = cur.fetchall()
    if not questions:
        print("⚠️ No interview questions stored")
        return

    per_q_scores = []
    for q_id, q_text in questions:
        cur.execute("SELECT interview_answer_text FROM interview_answer WHERE candidate_id=? AND iq_id=?", (candidate_id, q_id))
        row = cur.fetchone()
        if not row:
            print(f"⚠️ No answer for Q{q_id}")
            continue
        ans = row[0].strip()

        ans_skills = extract_skills_from_text(ans)
        kw_frac = keyword_overlap_score(jd_skills, ans_skills)
        sem_sim = semantic_similarity_score(jd_text, ans)
        llm_score, internet_ans = llm_evaluate_answer(jd_text, q_text, ans)
        sensible, doable = sensible_and_doable_flags(ans)
        final = aggregate_final_score(kw_frac, sem_sim, llm_score, sensible, doable)

        # --- UPSERT for candidate_score_detail ---
        cur.execute("""
            SELECT 1 FROM candidate_score_detail
            WHERE candidate_id=? AND jd_id=? AND question_id=?
        """, (candidate_id, jd_id, q_id))
        if cur.fetchone():
            cur.execute("""
                UPDATE candidate_score_detail
                SET keyword_overlap_pct=?, semantic_similarity=?, llm_score=?, final_score=?
                WHERE candidate_id=? AND jd_id=? AND question_id=?
            """, (kw_frac*100, sem_sim, llm_score, final, candidate_id, jd_id, q_id))
        else:
            cur.execute("""
                INSERT INTO candidate_score_detail
                (candidate_id, jd_id, question_id, keyword_overlap_pct, semantic_similarity, llm_score, final_score)
                VALUES (?,?,?,?,?,?,?)
            """, (candidate_id, jd_id, q_id, kw_frac*100, sem_sim, llm_score, final))
        conn.commit()

        per_q_scores.append(final)
        print(f"Q: {q_text}\n  kw:{kw_frac*100:.1f}% sem:{sem_sim} llm:{llm_score} final:{final} "
              f"(Sensible={sensible}, Doable={doable})")

    avg_score = round(sum(per_q_scores)/max(1,len(per_q_scores)), 2)

    # --- UPSERT for candidate_score_summary ---
    cur.execute("""
        SELECT 1 FROM candidate_score_summary
        WHERE candidate_id=? AND jd_id=?
    """, (candidate_id, jd_id))
    if cur.fetchone():
        cur.execute("""
            UPDATE candidate_score_summary
            SET overall_score=?
            WHERE candidate_id=? AND jd_id=?
        """, (avg_score, candidate_id, jd_id))
    else:
        cur.execute("""
            INSERT INTO candidate_score_summary (candidate_id,jd_id,overall_score)
            VALUES (?,?,?)
        """, (candidate_id, jd_id, avg_score))

    conn.commit()
    conn.close()
    update_best_candidates(db_path)
    print(f"\nInterview completed. Avg score={avg_score}")

# --------------------------
# Entry
# --------------------------

if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM candidate WHERE LOWER(is_scheduled)='y' LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        cid, name = row
        print(f"Running interview for {name} (ID={cid})")
        calculate_score(cid)
    else:
        print("No scheduled candidate found.")