"""
main.py
FastAPI backend. Flow per session:
1. Ask for name -> mobile -> email (stored in SQLite as a "lead")
2. Once all 3 are collected, switch to normal agent chat for the rest of the conversation

Run with: uvicorn main:app --reload --port 8000
"""

import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent  # noqa: E402
import db  # noqa: E402

app = FastAPI(title="AGC Admission Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this to your actual website domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

# Change this to something private before deploying - protects the admin panel
ADMIN_KEY = "changeme123"

# In-memory session store: {session_id: {"stage": ..., "lead": {...}, "history": [...]}}
sessions = {}

MOBILE_REGEX = re.compile(r"^\d{10}$")
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "stage": "name",
            "lead": {"name": None, "mobile": None, "email": None},
            "history": [],
        }
    return sessions[session_id]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    stage = session["stage"]
    text = req.message.strip()

    if stage == "name":
        if not text:
            return ChatResponse(reply="Please tell me your name to get started.")
        session["lead"]["name"] = text
        session["stage"] = "mobile"
        return ChatResponse(reply=f"Nice to meet you, {text}! Could you share your 10-digit mobile number?")

    if stage == "mobile":
        digits = re.sub(r"\D", "", text)
        if not MOBILE_REGEX.match(digits):
            return ChatResponse(reply="That doesn't look like a valid 10-digit mobile number. Please try again.")
        session["lead"]["mobile"] = digits
        session["stage"] = "email"
        return ChatResponse(reply="Great, and what's your email address?")

    if stage == "email":
        if not EMAIL_REGEX.match(text):
            return ChatResponse(reply="That doesn't look like a valid email address. Please try again.")
        session["lead"]["email"] = text
        session["stage"] = "chat"

        # Save the completed lead to the database
        db.save_lead(
            session_id=req.session_id,
            name=session["lead"]["name"],
            mobile=session["lead"]["mobile"],
            email=session["lead"]["email"],
        )

        name = session["lead"]["name"]
        return ChatResponse(
            reply=f"Thanks {name}! You're all set. Ask me anything about courses, fees, "
                  f"eligibility, deadlines, hostel, or scholarships."
        )

    # stage == "chat" -> normal agentic conversation
    reply, updated_history = run_agent(req.message, session["history"])
    session["history"] = updated_history
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/admin/leads")
def admin_leads(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"leads": db.get_all_leads()}