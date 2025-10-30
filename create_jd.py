import sqlite3
import pandas as pd
from flask import Flask, request, render_template_string
from threading import Thread
import os
from dotenv import load_dotenv

import time
import re

# ---- OpenAI >= 1.0 client ----
from openai import OpenAI

load_dotenv()

# Prefer standard OPENAI_*; fall back to your existing AZURE_* envs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")  # for Azure compat
# If you supplied an Azure endpoint root, add the compatibility path
if OPENAI_BASE_URL and OPENAI_BASE_URL.rstrip("/").endswith(".azure.com"):
    OPENAI_BASE_URL = OPENAI_BASE_URL.rstrip("/") + "/openai/v1"

# Model name (for Azure, use your deployment name here)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Build the client only if we have an API key
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if OPENAI_API_KEY else None

DB_FILE = "jd.db"

# -------------------- Database --------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Candidate table
    c.execute("""
    CREATE TABLE IF NOT EXISTS candidate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        jd_id INTEGER,
        name TEXT,
        email_id TEXT,
        contact_no TEXT,
        current_employer TEXT,
        current_ctc TEXT,
        gender TEXT,
        is_scheduled TEXT DEFAULT 'N',
        is_selected TEXT DEFAULT NULL,
        created_date DATE DEFAULT CURRENT_DATE
    )
    """)
    # Questions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        jd_id INTEGER,
        question_text TEXT,
        question_type TEXT,
        mandatory TEXT DEFAULT 'N'
    )
    """)
    # Answers table
    c.execute("""
    CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER,
        question_id INTEGER,
        answer_text TEXT
    )
    """)
    conn.commit()
    conn.close()

# -------------------- JD / Questions --------------------
MANDATORY_QUESTIONS = [
    ("Why are you interested in this role?", "behavioral", "Y"),
    ("Upload your resume (provide link if applicable)", "general", "N")
]

MAX_QUESTIONS_TOTAL = 10  # include mandatory ones

def _contains_long_jd_overlap(question: str, jd_text: str, window_words: int = 6) -> bool:
    """
    Return True if `question` contains a contiguous sequence of `window_words`
    that also appears in jd_text — used to detect model echoing the JD.
    """
    if not question or not jd_text or window_words <= 0:
        return False

    def _normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    q_norm = " " + _normalize(question) + " "
    jd_norm = _normalize(jd_text)
    jd_words = jd_norm.split()
    if len(jd_words) < window_words:
        return False

    for i in range(len(jd_words) - window_words + 1):
        chunk = " ".join(jd_words[i:i + window_words])
        if (" " + chunk + " ") in q_norm:
            return True
    return False

def generate_questions(requirements, candidate_profile: str = "", seniority_years: int = None):
    """
    Generalized LLM-driven generator for application/experience questions.
    - Balances pharma vs technical based on seniority.
    - Asks scenario-based questions only for high-use tools (Python, pandas, SQL).
    - No hardcoded fallback questions.
    """

    # 1. Determine remaining slots
    reserved = len(MANDATORY_QUESTIONS)
    remaining_slots = max(0, MAX_QUESTIONS_TOTAL - reserved)
    if remaining_slots == 0:
        return []

    # 2. Extract potential skills/keywords dynamically
    raw_tokens = []
    if requirements:
        parts = re.split(r'[,\|/()\n]', requirements)
        for p in parts:
            t = p.strip()
            if not t:
                continue
            if len(t.split()) <= 6:  # short phrases
                raw_tokens.append(t)

    skill_list_text = ", ".join(raw_tokens[:12]) if raw_tokens else ""

    # 3. Stage 1: extract structured insights
    stage1_prompt = f"""
You are an expert HR + Technical interviewer for pharma tech roles.
Analyze the context before producing questions.
Extract and structure:
1. Core technical skills (tools, languages, libraries, frameworks).
2. Pharma / domain-specific expertise.
3. Candidate seniority (if given); otherwise infer.
4. Guidance on balance:
   - 0–4 years: prioritize technical skills.
   - 5+ years: balance technical and pharma equally.
5. Map skills into potential question areas.
Do NOT generate questions yet, only reasoning.

Job description & recruiter notes:
{requirements}

Candidate profile:
{candidate_profile}
"""
    structured_insights = None
    try:
        stage1 = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": stage1_prompt}],
            max_tokens=512,
            temperature=0.3
        )
        structured_insights = stage1.choices[0].message.content.strip()
    except Exception as e:
        print("Stage 1 LLM call failed:", e)

    # 4. Stage 2: generate questions dynamically
    prompt = f"""
You are an expert interviewer crafting candidate-facing questions for pharma technical roles.
Do NOT produce coding tasks — only application/experience questions.

Context (structured + raw):
{structured_insights or ""}

Extracted key skills/terms: [{skill_list_text}]

Rules:
- Focus scenario-based questions only for most-used tools like Python, pandas, SQL.
- Other skills: general experience questions (years, exposure, comfort level).
- Balance pharma vs technical based on seniority (0–4 yrs → mostly technical, 5+ yrs → balanced).
- Questions should elicit concrete evidence (role, action, outcome, timeframe).
- Avoid marketing language and repetition from JD.
- Produce up to {remaining_slots} clear, open-ended questions.
- Return only the questions, one per line, no numbering.
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=480,
            temperature=0.45
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        print("Stage 2 LLM call failed:", e)
        return []

    # 5. Cleanup and filtering
    lines = [re.sub(r'^\s*[\-\•\*]?\s*\d{0,2}[\)\.\-]?\s*', '', ln.strip())
             for ln in content.splitlines() if ln.strip()]
    
    filtered = []
    skill_counts = {tok.lower(): 0 for tok in raw_tokens}

    for q in lines:
        if len(filtered) >= remaining_slots:
            break
        matched_skill = next((tok.lower() for tok in raw_tokens if tok.lower() in q.lower()), None)
        if matched_skill:
            if skill_counts.get(matched_skill, 0) >= 3:
                continue
            skill_counts[matched_skill] += 1
        filtered.append((q, "general", "N"))

    return filtered[:remaining_slots]



def create_local_form(jd_id, role):
    """Generate local form URL"""
    return f"http://127.0.0.1:5000/form/{jd_id}"

def create_jd(jd_record):
    """
    jd_record: dict with keys (as produced by SELECT * FROM job_description)
    Combine jd_text with user_agent_interaction text, generate questions, insert, update form_url.
    """
    jd_id = jd_record['id']
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    jd_text = jd_record.get('jd_text') or ""
    interaction_parts = []
    try:
        c.execute("SELECT user_request, clarifying_questions, user_responses FROM user_agent_interaction WHERE jd_id=?", (jd_id,))
        interactions = c.fetchall()
        for ur, cq, ursp in interactions:
            if ur:
                interaction_parts.append(str(ur))
            if cq:
                interaction_parts.append(str(cq))
            if ursp:
                interaction_parts.append(str(ursp))
    except Exception:
        interactions = []

    combined_text = jd_text
    if interaction_parts:
        combined_text = jd_text + ", " + ", ".join(interaction_parts)

    generated = generate_questions(combined_text)
    all_questions = []
    for q_text, q_type, q_mand in MANDATORY_QUESTIONS:
        all_questions.append((q_text, q_type, q_mand))
    for q in generated:
        if len(all_questions) >= MAX_QUESTIONS_TOTAL:
            break
        all_questions.append(q)

    for q_text, q_type, q_mand in all_questions:
        c.execute("SELECT 1 FROM questions WHERE jd_id=? AND question_text=?", (jd_id, q_text))
        if c.fetchone():
            continue
        c.execute("""
        INSERT INTO questions (jd_id, question_text, question_type, mandatory)
        VALUES (?, ?, ?, ?)
        """, (jd_id, q_text, q_type, q_mand))

    form_url = create_local_form(jd_id, jd_record.get('role'))
    c.execute("UPDATE job_description SET form_url=? WHERE id=?", (form_url, jd_id))
    conn.commit()
    conn.close()
    return form_url

# -------------------- Export to Excel --------------------
def export_to_excel():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM job_description", conn)
    df['Apply_Now'] = df['form_url']
    df.to_excel("job_descriptions.xlsx", index=False)
    conn.close()
    print("Exported job_descriptions.xlsx with Apply Now links")

# -------------------- Flask App --------------------
app = Flask(__name__)

FORM_HTML = """
<h2>Apply for {{role}}</h2>
<form method="POST">
  <label>Name: *</label><br><input type="text" name="name" required><br>
  <label>Email: *</label><br><input type="email" name="email_id" required><br>
  <label>Contact No: *</label><br><input type="text" name="contact_no" required><br>
  <label>Gender: *</label><br>
  <select name="gender" required>
    <option value="">--Select--</option>
    <option value="Male">Male</option>
    <option value="Female">Female</option>
    <option value="Other">Other</option>
  </select><br><br>
  <label>Current Employer:</label><br><input type="text" name="current_employer"><br>
  <label>Current CTC:</label><br><input type="text" name="current_ctc"><br>
  <hr>
  {% for q in questions %}
    <label>{{q[1]}} {% if q[2]=='Y' %}*{% endif %}</label><br>
    <input type="text" name="q{{q[0]}}" {% if q[2]=='Y' %}required{% endif %}><br><br>
  {% endfor %}
  <input type="submit" value="Submit">
</form>
"""

@app.route("/form/<int:jd_id>", methods=["GET", "POST"])
def form(jd_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role FROM job_description WHERE id=?", (jd_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return "JD not found", 404
    role = row[0]
    
    c.execute("SELECT id, question_text, mandatory FROM questions WHERE jd_id=?", (jd_id,))
    questions = c.fetchall()
    
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email_id")
        contact_no = request.form.get("contact_no")
        gender = request.form.get("gender")  # mandatory now
        current_employer = request.form.get("current_employer")
        current_ctc = request.form.get("current_ctc")

        c.execute("""
        INSERT INTO candidate (jd_id, name, email_id, contact_no, gender, current_employer, current_ctc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (jd_id, name, email, contact_no, gender, current_employer, current_ctc))
        candidate_id = c.lastrowid

        for q_id, q_text, mandatory in questions:
            ans = request.form.get(f"q{q_id}", "")
            c.execute("INSERT INTO answers (candidate_id, question_id, answer_text) VALUES (?, ?, ?)",
                    (candidate_id, q_id, ans))
        conn.commit()
        conn.close()
        return "Application submitted! Thank you."
    
    conn.close()
    return render_template_string(FORM_HTML, role=role, questions=questions)

def run_flask():
    app.run(debug=False, use_reloader=False)

# -------------------- Process Pending JDs --------------------
def process_pending_jds():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM job_description WHERE form_url IS NULL")
    pending = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
    conn.close()
    
    for jd in pending:
        form_url = create_jd(jd)
        print(f"JD {jd['id']} form created at {form_url}")

# -------------------- Main --------------------
if __name__ == "__main__":
    init_db()
    
    # 1. Process pending JDs (only those with form_url IS NULL)
    process_pending_jds()
    
    # 2. Export Excel
    export_to_excel()
    
    # 3. Start Flask server in background
    t = Thread(target=run_flask, daemon=True)  # daemon=True so it shuts down with main
    t.start()
    print("Local form server running at http://127.0.0.1:5000 ...")

    try:
        # Keep the main thread alive so CTRL+C works properly
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")