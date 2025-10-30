import os
import re
import json
import sqlite3
from dotenv import load_dotenv
from openai import AzureOpenAI
 
# ======================================================
# ENV & CLIENT (Azure OpenAI per your syntax)
# ======================================================
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT")
 
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
    # key is read automatically from AZURE_OPENAI_API_KEY
)
 
# ======================================================
# SQLITE SETUP (keeps your existing schema)
# ======================================================
DB_FILE = "jd.db"
 
def init_db():
    """Create tables if not exist for jobs and interaction logging."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_description (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            department TEXT,
            skills TEXT,
            experience TEXT,
            qualification TEXT,
            location TEXT,
            jd_text TEXT,
            form_url TEXT DEFAULT NULL,
            is_active TEXT DEFAULT 'Y',         -- default active
            is_fulfilled TEXT DEFAULT 'N',      -- default not fulfilled
            created_date DATE DEFAULT (DATE('now'))  -- default current date
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_agent_interaction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_request TEXT,
            clarifying_questions TEXT,
            user_responses TEXT,
            jd_id INTEGER,
            created_date DATE DEFAULT (DATE('now')),  -- default current date
            FOREIGN KEY (jd_id) REFERENCES job_description (id)
        )
    """)
    conn.commit()
    conn.close()
 
def clear_table():
    """Delete all rows (SQLite has no TRUNCATE)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM job_description;")
    conn.commit()
    conn.close()
    print("🧹 Cleared all rows from job_description.")
 
def save_to_db(role, department, skills, experience, qualification,  location, jd_text):
    """Insert a single JD into DB."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO job_description
        (role, department, skills, experience, qualification,  location, jd_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (role, department, skills, experience, qualification,  location, jd_text))
    conn.commit()
    jd_id = cursor.lastrowid
    conn.close()
    return jd_id
 
def save_interaction(user_request, clarifying_questions, user_responses, jd_id):
    """Save each recruiter interaction linked with JD ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_agent_interaction
        (user_request, clarifying_questions, user_responses, jd_id)
        VALUES (?, ?, ?, ?)
    """, (user_request, clarifying_questions, user_responses, jd_id))
    conn.commit()
    conn.close()
 
def view_jobs():
    """Print all rows for quick inspection."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, role, department, experience, qualification,location, jd_text FROM job_description ORDER BY id;")
    rows = cursor.fetchall()
    if not rows:
        print("ℹ️ No rows found.")
    for r in rows:
        print(r)
    conn.close()
 
def view_interactions():
    """Print all stored recruiter interactions."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_request, clarifying_questions, user_responses, jd_id FROM user_agent_interaction ORDER BY id;")
    rows = cursor.fetchall()
    if not rows:
        print("ℹ️ No interactions found.")
    for r in rows:
        print(r)
    conn.close()
 
# ======================================================
# MODEL HELPERS
# ======================================================
def _chat(prompt: str, temperature: float = 0.2) -> str:
    """Thin wrapper over Azure OpenAI chat.completions."""
    resp = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI recruiting assistant specializing in pharma technical roles"
                    "(data warehousing, reporting, analytics, data engineering, data management, Data Modelling, Quality Assurance,"
                    "Data preprocessing, Data Science, Data Architect, ETL Developer, Big Data Engineer). "
                    "Be crisp, structured,  Soft skills, Communication skills , prior pharma experience."
                    "Job Location Preference(Mumbai/Delhi/Bangalore/Pune/Hyderabad) and WFO Preference(Hybrid/Remote/In Office)"
                    "Strictly hire for the following Department/Lines of Business - Business Intelligence Management(BIM), Decision Science(DS), Commercial Excellence(CE), Research & Development(RND)"
                    "If department not mentioned ask. If user don't answer then do the prediction from the given set of LOBs/Departments."
                    "Convert experience into numerical range in years if provided in natural language, like 0-1 years(freshers), 5+ years(more than 5 years), 5-7 years(5 to 7 years)"
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()
 
def _extract_json(text: str):
    """
    Robustly extract JSON from model output.
    Handles cases with ```json fences or extra text.
    Returns a Python object (dict or list).
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if fence:
        block = fence.group(1)
        try:
            return json.loads(block)
        except Exception:
            pass
    generic = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if generic:
        block = generic.group(1)
        try:
            return json.loads(block)
        except Exception:
            pass
    raise ValueError("Could not parse JSON from model output.")
 
# ======================================================
# AGENT STEPS
# ======================================================
def ask_clarifying_questions(user_request: str) -> list[str]:
    """
    Ask up to 5 targeted clarification questions needed to create one or many JDs.
    Returns a list of question strings instead of a single string.
    """
    # --- Stage 1: Context extraction ---
    stage1_prompt = f"""
You are an AI assistant helping recruiters with pharma-technical job descriptions.

Step 1: Analyze the recruiter request.
Input request: "{user_request}"

Extract in structured notes:
- Role(s) mentioned
- Technical tools/skills explicitly stated
- Pharma domain context if any (commercial, clinical, quality, R&D, etc.)
- Department classification (BIM, DS, CE, RND) — predict if not explicit
- Experience normalization:
   * Fresher = 0-1 years
   * "5+" = more than 5 years
   * "5-7 years" = range 5 to 7
- Location preferences (if any)
- Work mode (Hybrid/Remote/In-office) if mentioned
- Missing info gaps
Return the above as a structured checklist (NOT questions yet).
"""
    try:
        structured_context = _chat(stage1_prompt, temperature=0.1)
    except Exception as e:
        print("Stage 1 context extraction failed:", e)
        structured_context = ""

    # --- Stage 2: Generate clarification questions ---
    stage2_prompt = f"""
The recruiter said: "{user_request}"

Context notes (from prior analysis):
{structured_context}

Now generate 2-5 clarifying questions:
- Do NOT repeat what is already clear from the request or notes.
- 70% must be about technical role details, tools, department alignment.
- 30% must be about location, work preference, qualification, and prior pharma experience.
- Always ensure experience is clarified in numeric years if unclear.
- For multiple roles, ask separate clarifications per role.
- Prefer concise, recruiter-friendly phrasing.
- Return only a numbered list of questions, one per line. No extra text.
"""
    raw_questions = _chat(stage2_prompt, temperature=0.2)

    # --- Parse the numbered list into Python list ---
    questions = []
    for line in raw_questions.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1. ", "2) ", etc.
        q = line
        if line[0].isdigit():
            q = line.lstrip('0123456789. )\t')
        questions.append(q.strip())
    
    return questions



def generate_structured_jds(user_request: str, recruiter_answers: str) -> list[dict]:
    """
    Generate one or multiple JDs based on the recruiter request and clarifications.
    Uses 2-step approach:
    - Stage 1: Normalize + enrich inputs (role, skills, dept, experience, pharma relevance).
    - Stage 2: Produce final JD(s) as JSON schema.
    """
    # --- Stage 1: Normalization ---
    stage1_prompt = f"""
Recruiter request:
\"\"\"{user_request}\"\"\"

Recruiter answers:
\"\"\"{recruiter_answers}\"\"\"

Step 1: Normalize and enrich the information.
- Identify roles, technical stack, pharma domain context.
- Normalize experience (fresher=0-1 years, etc.).
- Deduce department (BIM, DS, CE, RND).
- Predict missing details logically.
Return the notes in structured plain text.
"""
    try:
        structured_notes = _chat(stage1_prompt, temperature=0.1)
    except Exception as e:
        print("Stage 1 JD context enrichment failed:", e)
        structured_notes = ""

    # --- Stage 2: JD generation ---
    stage2_prompt = f"""
Recruiter request:
\"\"\"{user_request}\"\"\"

Recruiter answers:
\"\"\"{recruiter_answers}\"\"\"

Normalized notes:
{structured_notes}

Now produce FINAL job description(s) strictly in JSON.

Schema:
{{
  "jobs": [
    {{
      "role": "string",
      "department": "string",
      "skills": "comma-separated string",
      "experience": "string",
      "qualification": "string",
      "location": "comma-separated string",
      "jd_text": "string"
    }}
  ]
}}

Guidance:
- Reflect pharma context (commercial analytics, clinical data, manufacturing/quality).
- Include DW/ETL/reporting specifics (Snowflake, ADF, dbt, Airflow, Tableau/Power BI, SQL, Redshift, Databricks, PySpark, OLAP).
- For clouds (AWS, Azure, GCP) ensure consistency within ecosystem tools.
- Populate diverse but relevant tools in "skills".
- For team asks, create multiple roles (Data Engineer, BI Developer, Analyst, Architect, QA, Big Data Eng, Manager).
- Classify department into BIM, DS, CE, RND.
- Be concise but complete.
- No placeholders like TBD.
- Output ONLY JSON (no markdown fences, no explanation).
"""
    raw = _chat(stage2_prompt, temperature=0.2)
    data = _extract_json(raw)
    if isinstance(data, dict) and "jobs" in data and isinstance(data["jobs"], list):
        return data["jobs"]
    if isinstance(data, list):
        return data
    raise ValueError("Model did not return expected 'jobs' list.")

 
def insert_generated_jobs(jobs: list[dict], user_request, questions, answers):
    """Insert jobs and record interactions."""
    jd_ids = []
    for j in jobs:
        jd_id = save_to_db(
            role=j.get("role", ""),
            department=j.get("department", ""),
            skills=j.get("skills", ""),
            experience=j.get("experience", ""),
            qualification=j.get("qualification", ""),
            location=j.get("location", ""),
            jd_text=j.get("jd_text", "")
        )
        save_interaction(user_request, questions, answers, jd_id)
        jd_ids.append(jd_id)
    print(f"💾 Inserted {len(jobs)} job(s) into jd.db with interactions")
    return jd_ids
 
# ======================================================
# ORCHESTRATION
# ======================================================
def run_agent_interaction():
    """
    Full agent flow:
    1) Get free-form request
    2) Ask clarifying questions (one by one, sequentially)
    3) Collect answers interactively
    4) Generate JDs (one or many)
    5) Insert into DB
    6) Display a quick summary
    """
    user_request = input("💬 Enter your hiring request in natural language:\n> ").strip()
    if not user_request:
        print("❌ Empty request. Exiting.")
        return

    print("\n🤖 Clarifying questions:\n")
    questions = ask_clarifying_questions(user_request)

    answers_dict = {}
    for i, q in enumerate(questions, 1):
        ans = input(f"Q{i}: {q}\n> ").strip()
        answers_dict[q] = ans

    # Join answers in a consistent way for downstream JD generation
    combined_answers = "\n".join([f"{q} {a}" for q, a in answers_dict.items()])

    try:
        jobs = generate_structured_jds(user_request, combined_answers)
    except Exception as e:
        print(f"❌ Failed to generate structured JDs: {e}")
        return

    insert_generated_jobs(jobs, user_request, "\n".join(questions), combined_answers)

    print("\n📋 Stored roles:")
    for j in jobs:
        print(f" - {j.get('role','(no role)')} | Dept: {j.get('department','-')} | Exp: {j.get('experience','-')}")

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    init_db()
 
    # Simple interactive menu
    while True:
        print("\n=== JD Agent ===")
        print("1) Run agent (ask → answer → generate → store)")
        print("2) View stored JDs")
        print("3) View recruiter interactions")
        print("4) Clear table")
        print("5) Exit")
        choice = input("> ").strip()
 
        if choice == "1":
            run_agent_interaction()
        elif choice == "2":
            view_jobs()
        elif choice == "3":
            view_interactions()
        elif choice == "4":
            clear_table()
        elif choice == "5":
            break
        else:
            print("Please choose 1-5.")