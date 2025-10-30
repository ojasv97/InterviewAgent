#!/usr/bin/env python3

"""
interview_server.py

FastAPI backend for AI interview session with Azure OpenAI follow-up questions.
Supports multiple concurrent interviews using meeting_url routing.
Uses Web Speech API on client-side for transcription and Azure OpenAI for follow-up generation.
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import asyncio
from typing import Dict, Optional
import requests
import hashlib
from dotenv import load_dotenv

load_dotenv()
# ---------- CONFIG ----------
DB_PATH = os.getenv("JD_DB_PATH", "jd.db")
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_uploads")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Azure OpenAI config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_DEPLOYMENT_CHAT):
    print("Warning: Azure OpenAI credentials not found. Follow-up questions will be disabled.")
    print("Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_DEPLOYMENT_CHAT environment variables.")

# ---------- FastAPI ----------
app = FastAPI(title="AI Interview Server with Follow-ups")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def sql_connect():
    return sqlite3.connect(DB_PATH)

def fetch_schedule_by_meeting_url(conn, meeting_url: str):
    """Fetch schedule by meeting URL"""
    cur = conn.cursor()
    print(f"Looking for meeting_url: '{meeting_url}'")  # Debug line
    cur.execute("""SELECT id, jd_id, candidate_id, schedule_time, meeting_url, 
                status FROM interview_schedule WHERE status = 'scheduled' and meeting_url = ?""", (meeting_url,))
    r = cur.fetchone()
    if not r:
        print("No matching schedule found")  # Debug line
        return None
    cols = [d[0] for d in cur.description]
    result = dict(zip(cols, r))
    print(f"Found schedule: {result}")  # Debug line
    return result

def fetch_schedule(conn, schedule_id: str):
    cur = conn.cursor()
    cur.execute("SELECT id, jd_id, candidate_id, schedule_time, meeting_url, status FROM interview_schedule WHERE id = ?", (schedule_id,))
    r = cur.fetchone()
    if not r:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, r))

def fetch_candidate(conn, candidate_id: int):
    cur = conn.cursor()
    cur.execute("SELECT id, jd_id, name, email_id, contact_no, current_employer, current_ctc, gender FROM candidate WHERE id = ?", (candidate_id,))
    r = cur.fetchone()
    if not r:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, r))

def fetch_job_description(conn, jd_id: int):
    cur = conn.cursor()
    cur.execute("SELECT id, role, jd_text FROM job_description WHERE id = ?", (jd_id,))
    r = cur.fetchone()
    if not r:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, r))

def fetch_questions_for_schedule(conn, schedule_id: str):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, schedule_id, jd_id, candidate_id, interview_question_text, seq
        FROM interview_question
        WHERE schedule_id = ?
          AND parent_id IS NULL
        ORDER BY CAST(seq AS REAL)
    """, (schedule_id,))
    
    rows = cur.fetchall()
    return [
        {
            "iq_id": row[0],
            "schedule_id": row[1],
            "jd_id": row[2],
            "candidate_id": row[3],
            "text": row[4],
            "seq": row[5],
        }
        for row in rows
    ]

def upsert_interview_answer(conn, iq_id: str, schedule_id: str, candidate_id: int, transcript: str, audio_url: str, question_type: str = "main"):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interview_answer (iq_id, schedule_id, candidate_id, interview_answer_text, interview_answer_audio_url, created_date)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    """, (iq_id, schedule_id, candidate_id, transcript, audio_url))
    conn.commit()
    return cur.lastrowid

def generate_meeting_url(schedule_id: str) -> str:
    """Generate a unique meeting URL for a schedule"""
    hash_obj = hashlib.md5(schedule_id.encode())
    return f"/interview/{hash_obj.hexdigest()[:12]}"

# ---------- Azure OpenAI helper ----------
def azure_chat_completion_followups(jd_text: str, question: str, answer: str, max_followups: int = 3, timeout: int = 30):
    """
    Calls Azure GPT-4o-mini chat completion to generate follow-up questions
    """
    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_DEPLOYMENT_CHAT):
        print("Azure OpenAI not configured, skipping follow-up generation")
        return []
    
    api_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT_CHAT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

    system_message = (
        "You are an expert interviewer. Based on the candidate's answer, generate exactly 3 relevant follow-up questions "
        "that dig deeper into their response. The questions should be professional, specific, and help assess "
        "the candidate's skills and experience related to the job requirements. "
        "Return ONLY a valid JSON array of exactly 3 strings."
    )

    user_prompt = f"""
Job Description:
{jd_text}

Original Question:
{question}

Candidate's Answer:
{answer}

Generate exactly 3 follow-up questions based on this answer. Focus on:
1. Technical depth and specifics
2. Problem-solving approach
3. Real-world experience and examples

Return format: ["Question 1?", "Question 2?", "Question 3?"]
"""

    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            print(f"Azure chat completion failed ({resp.status_code}): {resp.text}")
            return []

        j = resp.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON response
        followups = json.loads(content)
        if isinstance(followups, list) and len(followups) >= 1:
            return followups[:max_followups]  # Limit to max_followups
        else:
            return []
    except Exception as e:
        print(f"Error generating follow-ups: {e}")
        return []

# ---------- Simple scoring function ----------
def simple_score_answer(jd_text: str, question: str, answer: str) -> float:
    """Simple keyword-based scoring"""
    if not answer or not answer.strip():
        return 0.0
    
    score = 5.0  # baseline
    
    # Length bonus
    words = len(answer.split())
    if words >= 30:
        score += 2.5
    elif words >= 20:
        score += 2.0
    elif words >= 10:
        score += 1.0
    elif words < 5:
        score -= 2.0
    
    # Keyword matching with job description
    if jd_text:
        jd_keywords = set(word.lower().strip('.,!?;:') for word in jd_text.split() if len(word) > 3)
        answer_keywords = set(word.lower().strip('.,!?;:') for word in answer.split() if len(word) > 3)
        common_keywords = jd_keywords.intersection(answer_keywords)
        score += min(len(common_keywords) * 0.5, 2.0)
    
    # Positive indicators
    positive_words = ['experience', 'project', 'team', 'developed', 'implemented', 'managed', 'years', 'worked', 'built', 'created']
    answer_lower = answer.lower()
    for word in positive_words:
        if word in answer_lower:
            score += 0.3
    
    # Technical terms bonus
    tech_words = ['python', 'javascript', 'database', 'api', 'framework', 'algorithm', 'testing', 'deployment']
    for word in tech_words:
        if word in answer_lower:
            score += 0.2
    
    return max(0.0, min(10.0, round(score, 1)))

# ---------- Session state tracking ----------
active_sessions: Dict[str, WebSocket] = {}  # schedule_id -> WebSocket
active_followups: Dict[str, Dict] = {}  # schedule_id -> {questions: [], followup_ids: [], index: int, parent_iq_id: int, parent_seq: int}

# ---------- HTTP Endpoints ----------
@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/api/session/{meeting_token}")
async def api_session_by_meeting(meeting_token: str):
    """Return candidate/session metadata for a given meeting URL token."""
    meeting_url = f"/interview/{meeting_token}"
    
    conn = sql_connect()
    try:
        sched = fetch_schedule_by_meeting_url(conn, meeting_url)
        if not sched:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cand = fetch_candidate(conn, sched["candidate_id"])
        jd = fetch_job_description(conn, sched["jd_id"])
        
        return JSONResponse({
            "schedule_id": sched["id"],
            "candidate_id": cand["id"] if cand else None,
            "candidate_name": cand.get("name") if cand else None,
            "candidate_gender": cand.get("gender", "M") if cand else "M",
            "role": jd.get("role") if jd else None,
            "meeting_url": sched["meeting_url"],
            "status": sched["status"]
        })
    finally:
        conn.close()

@app.get("/api/interviews/list")
async def list_all_interviews():
    """List all interview sessions"""
    conn = sql_connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.id, s.meeting_url, s.status, s.schedule_time, c.name, c.id as candidate_id, c.gender, j.role
            FROM interview_schedule s
            JOIN candidate c ON s.candidate_id = c.id
            JOIN job_description j ON s.jd_id = j.id
            ORDER BY s.schedule_time DESC
        """)
        
        interviews = []
        for row in cur.fetchall():
            interviews.append({
                "schedule_id": row[0],
                "meeting_url": row[1],
                "status": row[2],
                "schedule_time": row[3],
                "candidate_name": row[4],
                "candidate_id": row[5],
                "candidate_gender": row[6],
                "role": row[7],
                "full_url": f"http://localhost:8000{row[1]}"
            })
        
        return JSONResponse({"interviews": interviews})
    finally:
        conn.close()
async def send_websocket_message(session: str, message: dict):
    """Helper function to safely send WebSocket messages"""
    ws = active_sessions.get(session)
    if ws:
        try:
            await ws.send_json(message)
            print(f"[DEBUG] Sent WebSocket message: type={message.get('type')} to session={session}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send WebSocket message to session={session}: {e}")
            active_sessions.pop(session, None)
            return False
    else:
        print(f"[WARNING] No active WebSocket for session={session}")
        return False
@app.post("/api/upload_answer")
async def api_upload_answer(
    session: str = Form(...),
    iq_id: str = Form(...),  # now takes TEXT (will be numeric for main Qs, integer string for autoinc followups)
    elapsed_ms: float = Form(0.0),
    transcript: str = Form(""),
    file: UploadFile = File(...)
):
    """
    Receives audio file and transcript from client.
    Generates follow-up questions using Azure OpenAI after main questions.
    """
    conn = sql_connect()
    ans_id = None
    try:
        sched = fetch_schedule(conn, session)
        if not sched:
            print(f"[ERROR] Invalid session token: session={session}")
            raise HTTPException(status_code=404, detail="Invalid session token")
        
        candidate_id = sched["candidate_id"]
        jd = fetch_job_description(conn, sched["jd_id"]) or {}
        jd_text = jd.get("jd_text", "")

        # Save uploaded file as WAV (you said you want .wav always)
        filename = f"{uuid.uuid4()}.wav"
        local_path = os.path.join(AUDIO_DIR, filename)
        with open(local_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        
        audio_url = local_path
        
        if not transcript.strip():
            transcript = "No speech detected"
        
        # Normalize iq_id value
        iq_id = (iq_id or "").strip()

        # Debug: incoming values
        print(f"[DEBUG] upload_answer called. session={session} iq_id='{iq_id}' candidate_id={candidate_id}")

        # Determine if this is a follow-up flow already active for this session
        is_followup = session in active_followups
        
        if is_followup:
            # Handle follow-up answer
            af = active_followups[session]
            parent_iq_id = af["parent_iq_id"]
            idx = af["index"]
            
            # Validate follow-up index
            if idx >= len(af["followup_ids"]):
                print(f"[ERROR] Follow-up index {idx} out of range for session {session}")
                active_followups.pop(session, None)
                await send_next_main_question(conn, session)
                return JSONResponse({"ok": False, "error": "Invalid follow-up state"})
            
            followup_id = af["followup_ids"][idx]
            followup_question_text = af["questions"][idx]
            
            print(f"[DEBUG] Processing followup answer: idx={idx}, followup_id={followup_id}")

            # Save follow-up answer
            ans_id = upsert_interview_answer(conn, str(followup_id), session, candidate_id, transcript, audio_url, "followup")

            # Send answer confirmation
            payload = {
                "type": "answer_logged",
                "iq_id": str(followup_id),
                "parent_iq_id": parent_iq_id,
                "candidate_id": candidate_id,
                "question": followup_question_text,
                "transcript": transcript,
                "audio_url": audio_url,
                "elapsed_s": round((elapsed_ms or 0) / 1000.0, 2),
                "created_utc": datetime.utcnow().isoformat(),
                "question_type": "followup"
            }
            
            await send_websocket_message(session, payload)
            
            # Move to next follow-up
            af["index"] += 1
            
            if af["index"] < len(af["questions"]):
                # Send next follow-up
                next_followup = af["questions"][af["index"]]
                next_followup_id = af["followup_ids"][af["index"]]
                next_followup_seq = af["followup_seqs"][af["index"]]
                
                next_message = {
                    "type": "followup_question",
                    "iq_id": next_followup_id,
                    "text": next_followup,
                    "seq": next_followup_seq,
                    "followup_number": af["index"] + 1,
                    "total_followups": len(af["questions"]),
                    "question_type": "followup"  # CRITICAL: Include question_type
                }
                
                success = await send_websocket_message(session, next_message)
                if not success:
                    print(f"[ERROR] Failed to send next followup for session {session}")
                    active_followups.pop(session, None)
                    await send_next_main_question(conn, session)
            else:
                # Done with follow-ups, move to next main question
                print(f"[DEBUG] Completed all follow-ups for session={session}")
                active_followups.pop(session, None)
                await send_next_main_question(conn, session)

        else:
            # Handle main question answer
            cur = conn.cursor()

            # Validate iq_id for main question
            if not iq_id:
                print(f"[ERROR] Invalid iq_id for main question: iq_id='{iq_id}' session={session}")
                raise HTTPException(status_code=400, detail="Invalid iq_id for main question (empty). Ensure client sends iq_id in the upload request.")

            # convert to integer id for lookup
            try:
                main_iq_int = int(iq_id)
            except Exception:
                print(f"[ERROR] Invalid iq_id (not an integer) for main question: iq_id='{iq_id}' session={session}")
                raise HTTPException(status_code=400, detail="Invalid iq_id for main question (not an integer).")

            cur.execute("SELECT id, interview_question_text, seq FROM interview_question WHERE id = ?", (main_iq_int,))
            row = cur.fetchone()
            if not row:
                print(f"[ERROR] Interview question not found for id={main_iq_int}")
                raise HTTPException(status_code=404, detail="Interview question not found")
            
            question_text = row[1]
            question_seq = row[2]
            
            # Save main answer
            ans_id = upsert_interview_answer(conn, str(main_iq_int), session, candidate_id, transcript, audio_url, "main")
            print(f"[DEBUG] Main answer persisted: interview_answer.id={ans_id}, iq_id={main_iq_int}")

            # Notify client
            payload = {
                "type": "answer_logged",
                "iq_id": str(main_iq_int),
                "candidate_id": candidate_id,
                "question": question_text,
                "transcript": transcript,
                "audio_url": audio_url,
                "elapsed_s": round((elapsed_ms or 0) / 1000.0, 2),
                "created_utc": datetime.utcnow().isoformat(),
                "question_type": "main"
            }
            
            ws = active_sessions.get(session)
            if ws:
                try:
                    await ws.send_json(payload)
                except Exception:
                    active_sessions.pop(session, None)
            
            # Generate follow-up questions using Azure OpenAI (may return [])
            followups = azure_chat_completion_followups(jd_text or "", question_text, transcript)
            
            if followups:
                cur = conn.cursor()
                followup_ids = []
                followup_seqs = []
                # Insert follow-up rows (parent_id links back to main_iq_int)
                for i, fu_text in enumerate(followups, start=1):
                    fu_seq = f"{question_seq}.{i}"
                    
                    # Insert follow-up question as new row; parent_id stores parent main question id
                    cur.execute("""
                        INSERT INTO interview_question
                        (parent_id, schedule_id, jd_id, candidate_id, interview_question_text, seq)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (main_iq_int, session, sched["jd_id"], candidate_id, fu_text, fu_seq))
                    
                    # Get autoincremented id
                    fu_id = cur.lastrowid
                    followup_ids.append(fu_id)
                    followup_seqs.append(fu_seq)
                    print(f"[DEBUG] Inserted followup question: id={fu_id} seq={fu_seq} text={fu_text[:80]}")

                conn.commit()

                # Initialize follow-up tracking (store questions, ids, seqs)
                active_followups[session] = {
                    "questions": followups,
                    "followup_ids": followup_ids,
                    "followup_seqs": followup_seqs,
                    "index": 0,
                    "parent_iq_id": main_iq_int,
                    "parent_seq": question_seq
                }
                
                print("DEBUG: active_sessions keys:", list(active_sessions.keys()))
                print("DEBUG: sending followups for session (type,repr):", type(session), repr(session))

                # Send first follow-up (use seq like f"{question_seq}.1")
                ws = active_sessions.get(session)
                if ws:
                    try:
                        print(fu_id)
                        await ws.send_json({
                            "type": "followup_question",
                            "text": followups[0],
                            "seq": followup_seqs[0],
                            "followup_number": 1,
                            "iq_id": followup_ids[0],  # Use followup_ids[0] instead of fu_id
                            "total_followups": len(followups),
                            "question_type": "followup"  # ADD THIS LINE
                        })
                        print(f"DEBUG: followup_question sent for session={session} iq_id={followup_ids[0]}")
                    except Exception as e:
                        print("DEBUG: error sending followup_question:", e)
                        active_sessions.pop(session, None)
            else:
                print(f"DEBUG: no active websocket for session={session} (cannot push followup)")
                # No follow-ups generated, move to next main question
                await send_next_main_question(conn, session)
        
        # Final response
        return JSONResponse({
            "ok": True, 
            "transcript": transcript, 
            "audio_url": audio_url, 
            "answer_id": ans_id
        })
    finally:
        conn.close()

async def send_websocket_message(session: str, message: dict):
    """Helper function to safely send WebSocket messages"""
    ws = active_sessions.get(session)
    if ws:
        try:
            await ws.send_json(message)
            print(f"[DEBUG] Sent WebSocket message: type={message.get('type')} to session={session}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send WebSocket message to session={session}: {e}")
            active_sessions.pop(session, None)
            return False
    else:
        print(f"[WARNING] No active WebSocket for session={session}")
        return False
    
async def send_next_main_question(conn, session):
    """Enhanced version with better error handling and logging"""
    try:
        cur = conn.cursor()
        questions = fetch_questions_for_schedule(conn, session)
        total_main_questions = len(questions)
        # Get the last answered main question sequence
        cur.execute("""
            SELECT MAX(CAST(iq.seq AS REAL)) as max_seq
            FROM interview_answer ia
            JOIN interview_question iq ON CAST(ia.iq_id AS INTEGER) = iq.id
            WHERE ia.schedule_id = ? AND (iq.parent_id IS NULL OR iq.parent_id = '')
        """, (session,))
        
        last_row = cur.fetchone()
        last_seq = 0.0
        if last_row and last_row[0] is not None:
            try:
                last_seq = float(last_row[0])
            except Exception as e:
                print(f"[WARNING] Error parsing last_seq: {e}")
                last_seq = 0.0

        print(f"[DEBUG] Last answered main seq for session {session}: {last_seq}")

        # Get next main question
        cur.execute("""
            SELECT id, interview_question_text, seq
            FROM interview_question
            WHERE schedule_id = ? AND (parent_id IS NULL OR parent_id = '') 
            AND CAST(seq AS REAL) > ?
            ORDER BY CAST(seq AS REAL)
            LIMIT 1
        """, (session, last_seq))

        next_row = cur.fetchone()
        print(f"[DEBUG] Next main question for session {session}: {next_row}")

        if next_row:
            # Send next main question
            message = {
                "type": "question",
                "iq_id": str(next_row[0]),
                "text": next_row[1],
                "seq": next_row[2],
                "total_questions": total_main_questions,
                "question_type": "main"  # ADD THIS for consistency
            }
            
            success = await send_websocket_message(session, message)
            if not success:
                print(f"[ERROR] Failed to send next main question for session {session}")
        else:
            # No more questions - interview done
            completion_message = {"type": "done", "message": "Interview completed successfully"}
            await send_websocket_message(session, completion_message)
            
            # Mark schedule as completed
            cur.execute("UPDATE interview_schedule SET status = 'completed' WHERE id = ?", (session,))
            conn.commit()
            print(f"[DEBUG] Interview completed for session {session}")
            
    except Exception as e:
        print(f"[ERROR] Error in send_next_main_question for session {session}: {e}")
        # Try to send error message to client
        error_message = {"type": "error", "message": "An error occurred processing your request"}
        await send_websocket_message(session, error_message)


# ---------- WebSocket for candidate client ----------
@app.websocket("/ws/interview")
async def ws_interview(ws: WebSocket):
    """Candidate connects via WebSocket with ?session=<schedule_id> or ?meeting_token=<token>"""
    params = ws.query_params
    session = params.get("session")
    meeting_token = params.get("meeting_token")
    
    if not session and not meeting_token:
        await ws.close(code=1008)
        return
    
    # Fetch schedule
    
    conn = sql_connect()
    try:
        if meeting_token:
            meeting_url = f"/interview/{meeting_token}"
            sched = fetch_schedule_by_meeting_url(conn, meeting_url)
            if not sched:
                print(f"Schedule not found for meeting_url: {meeting_url}")
                await ws.close(code=1008)
                return
        else:
            sched = fetch_schedule(conn, session)
            if not sched:
                print(f"Schedule not found for session: {session}")
                await ws.close(code=1008)
                return

        # Normalize session id to string (important: use same type everywhere)
        session = str(sched["id"])

        print(f"WebSocket connected for schedule: {session}, candidate: {sched['candidate_id']}")
        
        cand = fetch_candidate(conn, sched["candidate_id"])
        jd = fetch_job_description(conn, sched["jd_id"])
        questions = fetch_questions_for_schedule(conn, session)
        total_main_questions = len(questions)
    finally:
        conn.close()
    
    # Register connection using the string session id
    await ws.accept()
    active_sessions[session] = ws
    
    # Short greeting (fixed) and immediately send the first question
    candidate_name = cand.get("name") if cand else "Candidate"
    candidate_gender = cand.get("gender", "M") if cand else "M"
    role_name = jd.get("role") if jd else "this position"
    
    greeting = f"Hi {candidate_name}, we'll interview you for the {role_name} role. Starting now."

    try:
        # send a compact greeting (clients can display / TTS it)...
        await ws.send_json({
            "type": "greeting",
            "text": greeting,
            "candidate_gender": candidate_gender
        })
    except Exception:
        pass

    # Send first question immediately (use dict keys from fetch_questions_for_schedule)
    if questions:
        first = questions[0]
        try:
            await ws.send_json({
                "type": "question",
                "iq_id": str(first["iq_id"]),
                "text": first["text"],
                "seq": first["seq"],
                "total_questions": total_main_questions
            })
        except Exception:
            pass
    else:
        try:
            print("Hey")
            await ws.send_json({"type": "done"})
        except Exception:
            pass
    
    # Keep connection alive and listen for client messages
    try:
        while True:
            msg_text = await ws.receive_text()
            try:
                m = json.loads(msg_text)
                if isinstance(m, dict) and m.get("type") == "bye":
                    await ws.close()
                    break
                # Handle other client messages if needed
            except json.JSONDecodeError:
                print(f"[WARNING] Invalid JSON from client in session {session}: {msg_text}")
                # Don't close connection for invalid JSON, just ignore
            except WebSocketDisconnect:
                print(f"[INFO] Client disconnected from session {session}")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error in WebSocket loop for session {session}: {e}")
                break
                
    except Exception as e:
        print(f"[ERROR] Error in WebSocket connection setup: {e}")
    finally:
        # Cleanup on disconnection
        if session:
            active_sessions.pop(session, None)
            active_followups.pop(session, None)
            print(f"[DEBUG] Cleaned up session {session}")
# ---------- Serve HTML client ----------
@app.get("/interview/{meeting_token}")
async def serve_interview_page(meeting_token: str):
    """Serve HTML interview page for a specific meeting token"""
    return FileResponse("interview_client.html")

@app.get("/")
async def serve_root():
    """Serve main page"""
    return FileResponse("interview_client.html")

# ---------- Initialize on startup ----------
@app.on_event("startup")
async def startup_event():
    print("AI Interview Server started successfully!")
    print(f"Audio files will be stored in: {AUDIO_DIR}")
    print(f"Database file: {DB_PATH}")
    if AZURE_OPENAI_API_KEY:
        print("Azure OpenAI integration: ENABLED")
    else:
        print("Azure OpenAI integration: DISABLED (set environment variables)")

# ---------- Run server ----------
if __name__ == "__main__":
    import uvicorn
    print("Starting AI Interview Server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)