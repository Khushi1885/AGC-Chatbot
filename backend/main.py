"""
main.py
FastAPI backend. Flow per session:
1. Ask for name -> mobile -> email (stored as a "lead")
2. Once all 3 are collected, switch to normal agent chat for the rest of the conversation
3. If the student mentions AGC NEST / scholarship, the bot offers to register them
   right in the chat (Yes/No buttons) and collects a few details, then emails
   them login credentials for the test portal.
4. For ambiguous course/fee questions (e.g. "btech fee"), the agent returns
   clickable branch options instead of guessing.

Run with: uvicorn main:app --reload --port 8000
"""

import random
import re
import string
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agent import run_agent  # noqa: E402
import db  # noqa: E402
import email_service  # noqa: E402
import lead_scoring  # noqa: E402

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

# Placeholder URL for the external AGC NEST test-taking site (build separately,
# then update this). The registered username is passed as a query param.
NEST_TEST_PORTAL_URL = "https://agcnest.netlify.app"

# Simple round-robin meeting slot picker (simulated scheduling - no real
# calendar integration). Rotates through business-hour slots across the next
# few business days so applications don't all get the exact same time.
MEETING_SLOTS = ["10:00 AM", "11:30 AM", "2:00 PM", "3:30 PM"]
_meeting_counter = {"count": 0}


def pick_next_meeting_slot():
    idx = _meeting_counter["count"]
    _meeting_counter["count"] += 1

    day_offset = idx // len(MEETING_SLOTS)  # which business day (0 = next business day)
    slot_time = MEETING_SLOTS[idx % len(MEETING_SLOTS)]

    target_date = datetime.now()
    business_days_added = -1
    while business_days_added < day_offset:
        target_date += timedelta(days=1)
        if target_date.weekday() < 5:  # Mon-Fri only
            business_days_added += 1

    return f"{target_date.strftime('%A, %d %b %Y')} at {slot_time}"

# In-memory session store
sessions = {}

MOBILE_REGEX = re.compile(r"^\d{10}$")
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

NEST_TRIGGER_KEYWORDS = [
    "agc nest", "nest test", "nest registration", "register for nest",
    "scholarship test", "scholarship exam", "apply for scholarship",
]
NEST_FIELDS = ["name", "email", "mobile", "course", "state", "category", "percentage"]
NEST_PROMPTS = {
    "name": "What's your full name for the AGC NEST registration?",
    "email": "What's your email address? (Your login details will be sent here.)",
    "mobile": "What's your 10-digit mobile number?",
    "course": "Which course are you interested in?",
    "state": "Which state are you from?",
    "category": "What's your category? (General / SC / ST / OBC)",
    "percentage": "What was your percentage in 12th (or your qualifying exam)?",
}


def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "stage": "name",
            "lead": {"name": None, "mobile": None, "email": None},
            "history": [],
            "nest": {"offered": False, "active": False, "step": 0, "data": {}, "last_username": None},
        }
    return sessions[session_id]


def generate_nest_credentials(name: str):
    first_name = re.sub(r"[^a-zA-Z]", "", name.split()[0]).lower() if name.strip() else "student"
    username = f"{first_name}{random.randint(100, 999)}"
    password = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return username, password


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    quick_replies: List[str] = []
    open_url: Optional[str] = None


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

        name = session["lead"]["name"]
        mobile = session["lead"]["mobile"]
        email = session["lead"]["email"]

        existing = db.find_existing_lead(mobile, email)
        if existing:
            return ChatResponse(
                reply=f"Looks like you've already chatted with us before, {name}! "
                      f"This mobile number or email is already registered. "
                      f"Go ahead and ask your question - I'm ready to help."
            )

        try:
            db.save_lead(session_id=req.session_id, name=name, mobile=mobile, email=email)
            print(f"[DB] Lead saved successfully: {session['lead']}")
        except Exception as e:
            print(f"[DB ERROR] Failed to save lead: {e}")

        return ChatResponse(
            reply=f"Thanks {name}! You're all set. Ask me anything about courses, fees, "
                  f"eligibility, deadlines, hostel, or scholarships."
        )

    # ---------------- stage == "chat" ----------------
    nest = session["nest"]

    # "Take Test Now" button -> open the external test portal
    if text.lower() == "take test now":
        username = nest.get("last_username")
        url = f"{NEST_TEST_PORTAL_URL}?user={username}" if username else NEST_TEST_PORTAL_URL
        return ChatResponse(reply="Opening the AGC NEST test portal for you...", open_url=url)

    # Actively collecting AGC NEST registration details, one field at a time
    if nest["active"]:
        field = NEST_FIELDS[nest["step"]]
        value = text

        if field == "mobile":
            digits = re.sub(r"\D", "", value)
            if not MOBILE_REGEX.match(digits):
                return ChatResponse(reply="That doesn't look like a valid 10-digit mobile number. Please try again.")
            value = digits
        if field == "email":
            if not EMAIL_REGEX.match(value):
                return ChatResponse(reply="That doesn't look like a valid email address. Please try again.")
        if field == "percentage":
            try:
                pct = float(re.sub(r"[^\d.]", "", value))
                if not (0 <= pct <= 100):
                    raise ValueError
                value = pct
            except ValueError:
                return ChatResponse(reply="Please enter a valid percentage (a number between 0 and 100).")

        nest["data"][field] = value
        nest["step"] += 1

        if nest["step"] < len(NEST_FIELDS):
            next_field = NEST_FIELDS[nest["step"]]
            return ChatResponse(reply=NEST_PROMPTS[next_field])

        # All fields collected - score the lead, generate credentials, save, email
        data = nest["data"]

        engagement = sum(1 for m in session["history"] if m.get("role") == "user")
        score, label = lead_scoring.predict_lead_score(
            percentage=data["percentage"], category=data["category"],
            course_name=data["course"], engagement=engagement, registered_nest=True,
        )
        print(f"[ML - Lead Evaluator] {data['name']}: {data['percentage']}% | engagement={engagement} "
              f"-> Score: {score}/100 -> {label.upper()} LEAD")

        username, password = generate_nest_credentials(data["name"])
        try:
            db.save_nest_registration(
                session_id=req.session_id, name=data["name"], email=data["email"],
                mobile=data["mobile"], course=data["course"], state=data["state"],
                category=data["category"], username=username, password=password,
                percentage=data["percentage"], lead_score=score, lead_label=label,
            )
            print(f"[DB] NEST registration saved: {username} ({data['email']})")
        except Exception as e:
            print(f"[DB ERROR] Failed to save NEST registration: {e}")
            nest["active"] = False
            return ChatResponse(reply="Something went wrong saving your registration. Please try again or contact the admission office at +91 8872009950.")

        test_url_for_email = f"{NEST_TEST_PORTAL_URL}?user={username}"
        email_service.send_nest_credentials(data["email"], data["name"], username, password, test_url_for_email)
        nest["active"] = False
        nest["offered"] = False
        nest["last_username"] = username

        reply = (
            f"You're registered for AGC NEST! Your login details have also been emailed to {data['email']}.\n\n"
            f"Username: {username}\nPassword: {password}\n\n"
            f"Click below when you're ready to take the test."
        )
        return ChatResponse(reply=reply, quick_replies=["Take Test Now"])

    # We just offered registration last turn - this message is the Yes/No answer
    if nest["offered"]:
        nest["offered"] = False
        lower = text.lower()
        if lower.startswith("yes"):
            nest["active"] = True
            nest["step"] = 0
            nest["data"] = {}
            return ChatResponse(reply=NEST_PROMPTS[NEST_FIELDS[0]])
        if lower.startswith("no"):
            return ChatResponse(reply="No problem! Let me know if you change your mind. Anything else about AGC I can help with?")
        # Anything else -> fall through to normal handling below

    # ---------------- Normal agentic conversation ----------------
    reply, updated_history, courses_discussed, quick_replies = run_agent(req.message, session["history"])
    session["history"] = updated_history

    for course_name in courses_discussed:
        try:
            if not db.was_course_already_logged(req.session_id, course_name):
                db.log_course_interest(
                    req.session_id, course_name,
                    mobile=session["lead"].get("mobile"),
                    email=session["lead"].get("email"),
                )
                print(f"[DB] Logged course interest: {course_name} for session {req.session_id}")
        except Exception as e:
            print(f"[DB ERROR] Failed to log course interest: {e}")

    # Offer AGC NEST registration if the student's message mentioned it
    if not nest["active"] and any(k in text.lower() for k in NEST_TRIGGER_KEYWORDS):
        nest["offered"] = True
        reply += "\n\nWould you like me to register you for AGC NEST right now?"
        quick_replies = ["Yes, register me", "No thanks"]

    return ChatResponse(reply=reply, quick_replies=quick_replies)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/admin/leads")
def admin_leads(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"leads": db.get_all_leads()}


class ApplicationRequest(BaseModel):
    name: str
    mobile: str
    email: str = ""
    program: str
    state: str = ""
    category: str = ""


class ApplicationResponse(BaseModel):
    success: bool
    message: str
    meeting_time: str = ""


@app.post("/apply", response_model=ApplicationResponse)
def apply(req: ApplicationRequest):
    meeting_time = pick_next_meeting_slot()

    try:
        db.save_application(req.name, req.mobile, req.email, req.program, req.state, req.category, meeting_time)
        print(f"[DB] Application saved: {req.name} - {req.program} - Meeting: {meeting_time}")
    except Exception as e:
        print(f"[DB ERROR] Failed to save application: {e}")
        return ApplicationResponse(
            success=False,
            message="We couldn't save your application right now. Please try again or call +91 8872009950.",
        )

    if req.email:
        email_service.send_application_confirmation(req.email, req.name, req.program, meeting_time)

    return ApplicationResponse(
        success=True,
        message="Application submitted successfully!",
        meeting_time=meeting_time,
    )


@app.get("/admin/applications")
def admin_applications(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"applications": db.get_all_applications()}


@app.get("/admin/nest-registrations")
def admin_nest_registrations(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"registrations": db.get_all_nest_registrations()}