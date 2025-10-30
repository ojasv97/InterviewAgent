# InterviewAgent
A completely automated AI agent for creating, scheduling and monitoring candidate interviews.

1. Project Overview
-------------------
This is a backend system for conducting AI-driven interviews.
It is built using FastAPI and SQLite for storing interview questions,
candidate sessions, and answers. It also includes frontend support
for recording and uploading audio responses. The project is modular,
with separate Python scripts for JD creation, goal gathering, scheduling, and scoring.

2. Project Structure
--------------------
project/
 ├── server.py             (Main FastAPI backend)
 ├── goal_gathering.py     (Handles goal collection logic)
 ├── create_jd.py          (Job description creation logic)
 ├── scheduler.py          (Interview scheduling logic)
 ├── scoring.py            (Answer scoring logic)
 ├── requirements.txt      (Python dependencies)
 ├── readme.txt            (This file)
 ├── jd.db                 (SQLite database)
 └── interview_client.html (Frontend files if any)

3. Python Modules Used
----------------------
- fastapi             : Web framework to create REST APIs
- uvicorn             : ASGI server to run FastAPI apps
- sqlite3             : Built-in Python DB library, stores questions & answers (db: jd.db)
- pydantic            : Validates and parses request/response models
- typing              : Type hints (List, Optional, Dict, etc.)
- uuid                : Generates unique session IDs
- logging             : For debug and server logs
- os                  : File and path operations
- wave                : Reading/writing WAV audio files
- vosk                : Speech recognition model for audio transcription
- threading           : For multithreaded operations (downloading/uploading)
- concurrent.futures  : ThreadPoolExecutor for concurrent tasks
- datetime            : Handling timestamps and durations
- time                : Time tracking for performance and elapsed time
- json                : JSON serialization and deserialization
- hashlib             : MD5 checks for data validation
- requests            : Making HTTP requests (if external APIs used)
- io                  : Input/Output streams (used for audio/files)
- traceback           : Error stack tracing
- contextlib          : Context managers (optional for resource handling)
- tempfile            : Temporary files handling

4. Setup Instructions
---------------------
Step 1: Open terminal and go to project directory
    cd project

Step 2: Create virtual environment
    python -m venv venv

Step 3: Activate environment
    - On Linux/Mac:
        source venv/bin/activate
    - On Windows (PowerShell):
        .\venv\Scripts\activate

Step 4: Install dependencies
    pip install -r requirements.txt

Step 5: Run the server
    uvicorn server:app --reload


5. Database: jd.db
----------------
This database stores all data related to the AI Interview System, including candidates, interview schedules, questions, answers, and scoring details.

Tables Overview
----------------

5.1. answers
-----------
- Stores all raw or processed answers submitted by candidates.
- Columns may include:
    - id (PK)
    - candidate_id (FK)
    - iq_id (FK to interview_question)
    - transcript
    - audio_file_path
    - elapsed_time
    - submission_timestamp

5.2. candidate
------------
- Stores information about candidates registered in the system.
- Columns may include:
    - id (PK)
    - name
    - email
    - phone
    - registration_date
    - status

5.3. candidate_score_detail
------------------------
- Stores detailed scoring for each answer provided by a candidate.
- Columns may include:
    - id (PK)
    - candidate_id (FK)
    - iq_id (FK)
    - score
    - remarks
    - evaluator_id
    - timestamp

5.4. candidate_score_summary
-------------------------
- Stores aggregate scores per candidate or session.
- Columns may include:
    - id (PK)
    - candidate_id (FK)
    - total_score
    - average_score
    - session_id (FK to interview_schedule)
    - evaluation_date

5.5. interview_answer
------------------
- Stores answers for interview questions (similar to `answers`, may include historical or raw data).
- Columns may include:
    - id (PK)
    - candidate_id (FK)
    - iq_id (FK)
    - transcript
    - audio_file_path
    - status
    - timestamp

5.6. interview_question
---------------------
- Stores interview questions, including follow-ups and sequencing.
- Columns may include:
    - id (PK)
    - schedule_id (FK to interview_schedule)
    - parent_id (nullable FK to self)
    - jd_id (FK to job_description)
    - candidate_id (FK)
    - interview_question_text
    - seq
    - iq_type (main/followup)

5.7. interview_schedule
---------------------
- Stores interview schedules for candidates.
- Columns may include:
    - id (PK)
    - candidate_id (FK)
    - schedule_date
    - start_time
    - end_time
    - status

5.8. job_description
------------------
- Stores job descriptions linked to interview questions.
- Columns may include:
    - id (PK)
    - title
    - description_text
    - required_skills
    - created_date
    - created_by

5.9. questions
------------
- May store a master list of questions available in the system.
- Columns may include:
    - id (PK)
    - question_text
    - category
    - difficulty_level
    - created_date

5.10. user_agent_interaction
--------------------------
- Logs user interactions with the system (frontend events, API usage).
- Columns may include:
    - id (PK)
    - candidate_id (FK)
    - session_id (FK)
    - action
    - timestamp
    - metadata (JSON)

6. Features
-----------
- Candidate session management
- Fetch interview questions per schedule
- Upload and validate answers
- Audio recording and Vosk transcription
- Main and follow-up question support
- Goal gathering workflow
- Job description creation and management
- Interview scheduling automation
- Answer scoring and evaluation
- Multithreaded processing for downloads/uploads
- MD5-based data validation
- JSON-based request/response handling

=============================
 End of README
=============================
