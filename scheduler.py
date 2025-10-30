#!/usr/bin/env python3
"""
scheduler.py

Schedules interviews for candidates with is_scheduled = 'N'.
"""
import os
import sqlite3
import requests
import json
import re
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
DB_PATH = os.getenv("JD_DB_PATH", "jd.db")
LOCAL_INTERVIEW_BASE = os.getenv("LOCAL_INTERVIEW_BASE", "/interview")
SCHEDULE_OFFSET_HOURS = int(os.getenv("SCHEDULE_OFFSET_HOURS", "24"))

# ---- Azure OpenAI config (from your .env) -----
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT")
AZURE_API_VERSION = "2024-02-15-preview"

if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_DEPLOYMENT_CHAT):
    raise RuntimeError("Azure OpenAI API environment variables must be set.")

# ---------- Utilities ----------
def connect():
    return sqlite3.connect(DB_PATH)

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _contains_long_jd_overlap(question: str, jd_text: str, window_words: int = 6) -> bool:
    if not question or not jd_text or window_words <= 0:
        return False
    qnorm = " " + _normalize_text(question) + " "
    jd_norm = _normalize_text(jd_text)
    jd_words = jd_norm.split()
    if len(jd_words) < window_words:
        return False
    for i in range(len(jd_words) - window_words + 1):
        chunk = " ".join(jd_words[i:i+window_words])
        if (" " + chunk + " ") in qnorm:
            return True
    return False

def _extract_short_tokens(text: str, max_tokens: int = 24):
    if not text:
        return []
    parts = re.split(r'[,\|/()\n;:]', text)
    tokens = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        if 1 <= len(t.split()) <= 4 and not t.isdigit():
            tokens.append(t)
    seen, out = set(), []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
        if len(out) >= max_tokens:
            break
    return out

# ---------- DB init ----------
def init_interview_tables(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interview_schedule (
        id TEXT PRIMARY KEY,
        jd_id INTEGER,
        candidate_id INTEGER,
        schedule_time TEXT,
        meeting_url TEXT,
        meeting_start_time TEXT DEFAULT NULL,
        meeting_end_time TEXT DEFAULT NULL,
        status TEXT DEFAULT 'scheduled',
        created_date TEXT DEFAULT (DATETIME('now')),
        FOREIGN KEY (jd_id) REFERENCES job_description (id),
        FOREIGN KEY (candidate_id) REFERENCES candidate (id)                
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interview_question (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        jd_id INTEGER,                                -- job description reference
        schedule_id TEXT,                             -- links to interview_schedule
        candidate_id INTEGER,                         -- candidate reference
        parent_id INTEGER,                            -- NULL = main Q, else FK to main Q
        interview_question_text TEXT NOT NULL,                  -- actual question text
        seq INTEGER,                                  -- order of the question
        created_date TEXT DEFAULT (DATETIME('now')),  -- timestamp
        -- FKs
        FOREIGN KEY (jd_id) REFERENCES job_description (id),
        FOREIGN KEY (schedule_id) REFERENCES interview_schedule (id),
        FOREIGN KEY (candidate_id) REFERENCES candidate (id),
        FOREIGN KEY (parent_id) REFERENCES interview_question (id) -- self-reference
    )   
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS interview_answer (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        iq_id INTEGER,
        schedule_id TEXT,
        candidate_id INTEGER,
        interview_answer_text TEXT,
        interview_answer_audio_url TEXT,
        created_date TEXT DEFAULT (DATETIME('now')),
        FOREIGN KEY (iq_id) REFERENCES interview_question (id),
        FOREIGN KEY (candidate_id) REFERENCES candidate (id),
        FOREIGN KEY (schedule_id) REFERENCES interview_schedule (id)
    )
    """)
    conn.commit()

# ---------- LLM (REST) ----------
def call_openai_responses(prompt: str, timeout: int = 60) -> str:
    api_url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_DEPLOYMENT_CHAT}/chat/completions?api-version={AZURE_API_VERSION}"
    )
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert interview designer."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.15,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Azure OpenAI error {resp.status_code}: {err}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ---------- Data fetch ----------
def fetch_pending_candidates(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM candidate WHERE is_scheduled = 'N'")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def fetch_job_description(conn, jd_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM job_description WHERE id = ?", (jd_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))

def fetch_user_agent_interactions(conn, jd_id):
    cur = conn.cursor()
    cur.execute("SELECT user_request, clarifying_questions, user_responses FROM user_agent_interaction WHERE jd_id = ?", (jd_id,))
    return cur.fetchall()

def fetch_candidate_form_answers(conn, candidate_id):
    cur = conn.cursor()
    cur.execute("""
    SELECT q.question_text, a.answer_text
    FROM answers a
    JOIN questions q ON a.question_id = q.id
    WHERE a.candidate_id = ?
    ORDER BY q.id
    """, (candidate_id,))
    return cur.fetchall()

# ---------- Prompt & generation (unchanged) ----------
# ---------- Prompt builder ----------
def build_breadth_prompt(jd_text: str, interactions: list, candidate_answers: list, max_questions: int = 10):
    interaction_text = ""
    if interactions:
        parts = []
        for ur, cq, ursp in interactions:
            if ur:
                parts.append(f"recruiter_request: {ur}")
            if cq:
                parts.append(f"clarifying_questions: {cq}")
            if ursp:
                parts.append(f"recruiter_answers: {ursp}")
        interaction_text = "\n".join(parts)

    cand_text = ""
    if candidate_answers:
        parts = []
        for qtxt, atxt in candidate_answers:
            q_clean = qtxt.strip().replace("\n", " ")
            a_clean = (atxt or "").strip().replace("\n", " ")
            parts.append(f"Q: {q_clean}\nA: {a_clean}")
        cand_text = "\n\n".join(parts)

    tokens = _extract_short_tokens(jd_text + "\n" + interaction_text)
    token_list = ", ".join(tokens[:12])

    prompt = f"""
You are an expert interview designer. Produce a set of interview questions for a structured 1:1 AI-led screening session.
Context: Job description (below), recruiter interactions, and candidate's submitted form answers (if any).
Job description:
{jd_text}

Recruiter interactions (if present):
{interaction_text or '(none)'}

Candidate submitted answers (if present):
{cand_text or '(none)'}

Detected skill/tool hints: [{token_list}]

Requirements & rules:
1. Generate up to {max_questions} concise questions that prioritize breadth: cover as many different skills/topics mentioned in the JD and candidate's answers as possible. Do NOT dive into lengthy, multi-step coding problems – depth will be explored during live conversation.
2. For each question include a short category: 'technical' (skill/tool/experience), 'behavioral' (situation/result), or 'general' (logistics, resume, availability).
3. For technical questions, prefer precise, measurable phrasing that elicits facts such as years, versions, frequency, scale, or concrete outcomes. Example good forms:
   - "How many years of hands-on experience do you have with Python?"
   - "Which versions of Snowflake have you used in production?"
   - "How often do the ETL jobs you maintain run?"
   - "Rate your SQL proficiency (Beginner/Intermediate/Advanced) and provide one-line evidence."
4. For candidate-provided answers present in the context, craft clarifying breadth questions that convert vague answers into measurable items. Example: if candidate wrote 'worked on pipelines', ask "What was the typical input size (rows/GB) and runtime for the pipelines you worked on?"
5. Avoid repeating sentences verbatim from the job description. Avoid prompts like "Explain your experience with We are seeking...".
6. Aim to *cover* different areas (ETL, cloud, data warehousing, SQL, Python, BI) rather than deeply probing a single area. At most 3 questions should mention the same explicit skill token.
7. Output format: JSON array only. Each item must be an object with keys: "question" (string), "category" (one of "technical","behavioral","general").
8. Return only valid JSON — nothing else.

Now produce the JSON array of questions.
"""
    return prompt

# ---------- Generate interview questions via LLM ----------
def generate_interview_questions(jd_text: str, interactions: list, candidate_answers: list, max_q: int = 10):
    prompt = build_breadth_prompt(jd_text, interactions, candidate_answers, max_questions=max_q)
    raw = call_openai_responses(prompt)
    start = raw.find('[')
    end = raw.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("LLM did not return a JSON array of questions.")
    json_text = raw[start:end+1]
    try:
        obj = json.loads(json_text)
        if not isinstance(obj, list):
            raise RuntimeError("LLM returned JSON but not a list.")
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM response: {e}\nraw output:\n{raw}")
    questions = []
    token_counts = {}
    tokens = _extract_short_tokens(jd_text)
    for item in obj:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        cat = item.get("category", "general")
        if not q or not isinstance(q, str):
            continue
        if _contains_long_jd_overlap(q, jd_text, window_words=6):
            continue
        if len(q.split()) < 4:
            continue
        matched = None
        q_l = q.lower()
        for tok in tokens:
            if tok.lower() in q_l:
                matched = tok.lower()
                break
        if matched:
            token_counts.setdefault(matched, 0)
            if token_counts[matched] >= 3:
                continue
            token_counts[matched] += 1
        questions.append((q.strip(), cat if cat in ("technical","behavioral","general") else "general"))
        if len(questions) >= max_q:
            break

    if not questions:
        raise RuntimeError("After filtering, no valid interview questions were produced by the LLM.")
    return questions


# ---------- Scheduling ----------
def create_local_meeting_url(schedule_id: str) -> str:
    return f"{LOCAL_INTERVIEW_BASE}/{schedule_id}"

def schedule_candidate(conn, candidate_row):
    cur = conn.cursor()
    candidate_id = candidate_row["id"]
    jd_id = candidate_row.get("jd_id") or candidate_row.get("job_id") or candidate_row.get("job_description_id")
    if not jd_id:
        raise RuntimeError(f"Candidate {candidate_id} missing jd_id/job reference.")
    jd = fetch_job_description(conn, jd_id)
    jd_text = jd.get("jd_text", "") if jd else ""
    interactions = fetch_user_agent_interactions(conn, jd_id)
    candidate_answers = fetch_candidate_form_answers(conn, candidate_id)

    schedule_id = str(uuid.uuid4())
    schedule_time = (datetime.utcnow() + timedelta(hours=SCHEDULE_OFFSET_HOURS)).isoformat()
    meeting_url = create_local_meeting_url(schedule_id)

    questions = generate_interview_questions(jd_text, interactions, candidate_answers, max_q=10)

    # insert schedule
    cur.execute("""
    INSERT INTO interview_schedule (id, jd_id, candidate_id, schedule_time, meeting_url, status)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (schedule_id, jd_id, candidate_id, schedule_time, meeting_url, 'scheduled'))

    seq = 1
    for q_text, category in questions:
        # insert into interview_question
        cur.execute("""
        INSERT INTO interview_question (parent_id, schedule_id, jd_id, candidate_id, interview_question_text, seq)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (None, schedule_id, jd_id, candidate_id, q_text, seq))
        iq_id = cur.lastrowid
        seq += 1

    cur.execute("UPDATE candidate SET is_scheduled = 'Y' WHERE id = ?", (candidate_id,))
    conn.commit()
    return schedule_id, meeting_url, len(questions)

# ---------- Orchestration ----------
def process_pending_candidates():
    conn = connect()
    try:
        init_interview_tables(conn)
        pending = fetch_pending_candidates(conn)
        if not pending:
            print("No candidates with is_scheduled = 'N'.")
            return
        for cand in pending:
            try:
                sid, url, qcount = schedule_candidate(conn, cand)
                print(f"Scheduled candidate id={cand['id']} jd_id={cand.get('jd_id')} schedule={sid} url={url} questions={qcount}")
            except Exception as e:
                print(f"ERROR scheduling candidate id={cand.get('id')}: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    process_pending_candidates()