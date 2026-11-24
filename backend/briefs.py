"""
briefs.py
1) Saves each chatbot turn to the 'chat_messages' table, so the conversation
   survives server restarts and counsellors can be briefed from it.
2) Builds a short, 3-line counsellor brief for every assigned student:

     Line 1  FACTS     - built in Python from stored data (course, %, category, lead label...)
     Line 2  TOPICS    - detected by keyword rules from the student's own messages
     Line 3  AI NOTE   - 1-2 sentence summary written by the local LLM

   The model never supplies numbers. Its summary is also verified: if it contains
   any number that does not appear in the transcript, it is discarded and a safe
   rule-based line is used instead. If Ollama is down, the brief still works.
"""

import re
import requests

from db import supabase
import counsellors

TOPIC_KEYWORDS = [
    ("fees", ["fee", "fees", "cost", "tuition", "charges", "kitni"]),
    ("hostel", ["hostel", "mess", "accommodation", "room"]),
    ("scholarship", ["scholarship", "nest", "waiver", "tfw", "pms", "concession"]),
    ("placements", ["placement", "package", "recruit", "salary", "job"]),
    ("eligibility / admission process", ["eligib", "criteria", "admission process", "deadline",
                                         "last date", "counselling", "counseling", "cutoff"]),
    ("international programs", ["international", "abroad", "overseas", "exchange"]),
]


def _ollama_config():
    try:
        from agent import OLLAMA_URL, MODEL_NAME
        return OLLAMA_URL, MODEL_NAME
    except Exception:
        return "http://localhost:11434/api/chat", "qwen3:1.7b"


# ---------------------------------------------------------------- chat persistence

def save_chat_turn(session_id: str, user_text: str, assistant_text: str):
    supabase.table("chat_messages").insert([
        {"session_id": session_id, "role": "user", "content": (user_text or "")[:2000]},
        {"session_id": session_id, "role": "assistant", "content": (assistant_text or "")[:2000]},
    ]).execute()


def _find_session_ids(mobile: str, email: str):
    """The leads table links a student's mobile/email to the chat session(s) they used."""
    ids = set()
    m, e = counsellors._norm_mobile(mobile), counsellors._norm_email(email)
    if m:
        for r in supabase.table("leads").select("*").eq("mobile", m).execute().data:
            ids.add(r["session_id"])
    if e:
        for r in supabase.table("leads").select("*").eq("email", e).execute().data:
            ids.add(r["session_id"])
    return ids


def get_chat_for_student(mobile: str, email: str):
    messages, session_ids = [], _find_session_ids(mobile, email)
    for sid in session_ids:
        messages += supabase.table("chat_messages").select("*").eq("session_id", sid).execute().data
    messages.sort(key=lambda r: (r.get("created_at") or "", r.get("id", 0)))
    return messages, session_ids


# ---------------------------------------------------------------- brief building

def detect_topics(user_messages):
    text = " ".join(user_messages).lower()
    return [name for name, words in TOPIC_KEYWORDS if any(w in text for w in words)]


def _courses_discussed(session_ids):
    names = []
    for sid in session_ids:
        try:
            for r in supabase.table("course_interest").select("*").eq("session_id", sid).execute().data:
                if r["course_name"] not in names:
                    names.append(r["course_name"])
        except Exception:
            pass
    return names


def _facts_line(a: dict) -> str:
    kind = "AGC NEST registrant" if a.get("source") == "nest" else "admission inquiry"
    line = f"{a.get('student_name')} - {a.get('course') or 'course not specified'} ({kind})."
    bits = []
    if a.get("percentage") is not None:
        bits.append(f"12th: {float(a['percentage']):g}%")
    if a.get("category"):
        bits.append(f"Category: {a['category']}")
    if a.get("state"):
        bits.append(f"State: {a['state']}")
    if a.get("lead_label"):
        conf = f" ({float(a['lead_score']):g}% model confidence)" if a.get("lead_score") is not None else ""
        bits.append(f"Lead: {a['lead_label']}{conf}")
    return line + (" " + " | ".join(bits) if bits else "")


def _numbers(text: str):
    return set(re.findall(r"\d+(?:[.,]\d+)?", text or ""))


def _grounded(summary: str, transcript: str) -> bool:
    """Verifier: every number in the AI summary must already be in the transcript."""
    return _numbers(summary) <= _numbers(transcript)


def _clean_llm_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    text = re.sub(r"[*_#`>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()[:400]


def _llm_summary(messages) -> str:
    """Returns a verified 1-2 sentence summary, or '' if unavailable/unsafe."""
    recent = messages[-12:]
    if not any(m["role"] == "user" for m in recent):
        return ""
    transcript = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}" for m in recent
    )
    url, model = _ollama_config()
    try:
        resp = requests.post(url, json={
            "model": model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 120},
            "messages": [
                {"role": "system", "content": (
                    "You write a short internal note for a college admission counsellor, based ONLY on "
                    "the chat transcript. Write at most 2 short sentences about what the student wants "
                    "or is worried about. Never invent facts, numbers, fees or promises. "
                    "Output plain text only.")},
                {"role": "user", "content": transcript},
            ],
        }, timeout=60)
        summary = _clean_llm_text(resp.json()["message"]["content"])
    except Exception as e:
        print(f"[BRIEF] LLM summary unavailable: {e}")
        return ""
    if not summary or not _grounded(summary, transcript):
        print("[BRIEF] LLM summary rejected by verifier (empty or ungrounded numbers)")
        return ""
    return summary


def build_brief(assignment: dict) -> str:
    messages, session_ids = get_chat_for_student(assignment.get("student_mobile"), assignment.get("student_email"))
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]

    line1 = _facts_line(assignment)

    if user_msgs:
        topics = detect_topics(user_msgs)
        courses = _courses_discussed(session_ids)
        parts = []
        if topics:
            parts.append("Asked about: " + ", ".join(topics) + ".")
        if courses:
            parts.append("Courses discussed: " + ", ".join(courses) + ".")
        line2 = " ".join(parts) or f"Chatted with the assistant ({len(user_msgs)} messages), no specific topic detected."
    else:
        line2 = "No chatbot conversation linked to this student (they used the form directly)."

    summary = _llm_summary(messages)
    if summary:
        line3 = "AI note: " + summary
    else:
        slot = assignment.get("meeting_time")
        line3 = "Next step: " + (f"call the student for the counselling slot - {slot}." if slot
                                 else "contact the student and fix a counselling slot.")
    return "\n".join([line1, line2, line3])


def generate_and_store_brief(assignment_id: int):
    """Safe to run in a background task. Returns the brief text, or None on failure."""
    try:
        assignment = counsellors.get_assignment(assignment_id)
        if not assignment:
            return None
        brief = build_brief(assignment)
        counsellors.set_brief(assignment_id, brief)
        print(f"[BRIEF] Stored brief for assignment {assignment_id}")
        return brief
    except Exception as e:
        print(f"[BRIEF ERROR] Could not generate brief for assignment {assignment_id}: {e}")
        return None