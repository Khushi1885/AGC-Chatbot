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

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agent import run_agent  # noqa: E402
import db  # noqa: E402
import email_service  # noqa: E402
import lead_scoring  # noqa: E402
import counsellors  # noqa: E402
import briefs  # noqa: E402
import admin_agent  # noqa: E402

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

# Explicit "please register me now" intent - skips the Yes/No offer step
DIRECT_REGISTRATION_KEYWORDS = [
    "register me for agc nest", "register me for nest", "i want to register for agc nest",
    "i want to register for nest", "register for agc nest now", "sign me up for agc nest",
    "sign me up for nest", "nest me register", "nest ke liye register",
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


# ---------------- Counselling slot booking / rescheduling (chat) ----------------
# Deterministic on purpose: the model is not trusted to book or move real appointments.
# Python detects the intent, reads the counsellor's real availability, offers options as
# buttons, and books the one the student taps.
SLOT_STRONG_TRIGGERS = [
    "reschedule", "re-schedule", "change my slot", "change slot", "change my counselling",
    "change my counseling", "change my call", "change my meeting", "change the time",
    "my counselling slot", "my counseling slot", "my slot", "book a slot", "book my slot",
    "book counselling", "book counseling", "slot badal", "time badal",
]
SLOT_WEAK_TRIGGERS = [
    "possible nahi", "possible nhi", "not possible", "can't make", "cant make", "cannot make",
    "meeting time", "call time",
]
SLOT_CANCEL_WORDS = {"cancel", "never mind", "nevermind", "no", "no thanks", "nahi", "skip"}


def _handle_slot_request(session, text, chosen, strong):
    lead = session["lead"]
    assignment = counsellors.find_assignment_for_student(lead.get("mobile"), lead.get("email"))
    if not assignment:
        if not strong and not chosen:
            return None     # weak trigger and no counselling assigned -> let the normal agent answer
        session["slot"]["options"] = []
        return ChatResponse(
            reply="I couldn't find a counselling assignment for you yet. Submit the admission inquiry "
                  "form or register for AGC NEST - a counsellor is allocated automatically, and then I "
                  "can book a slot for you."
        )
    counsellor = counsellors.get_counsellor(assignment["counsellor_id"])
    if not counsellor:
        return ChatResponse(reply="I couldn't reach your counsellor's calendar right now. "
                                  "Please call the admission office at +91 8872009950.")

    prefix = ""
    if chosen:
        result = counsellors.book_slot(assignment, chosen["date"], chosen["time"])
        if result["ok"]:
            session["slot"]["options"] = []
            try:
                email_service.send_slot_confirmation(
                    lead.get("email"), lead.get("name") or assignment.get("student_name"),
                    result["label"], counsellor)
            except Exception as e:
                print(f"[EMAIL ERROR] Slot confirmation failed: {e}")
            return ChatResponse(
                reply=f"Done! Your counselling call with {counsellor['name']} is booked for "
                      f"{result['label']}. I've emailed you the confirmation."
            )
        prefix = "Sorry, that slot just became unavailable. "

    options, _prefs = counsellors.suggest_slots(assignment["counsellor_id"], "" if chosen else text)
    session["slot"]["options"] = options
    if not options:
        return ChatResponse(
            reply=prefix + "I couldn't find a free slot matching that. Try another day "
                  "(e.g. 'Tuesday') or time of day ('morning' / 'afternoon').")
    current = assignment.get("meeting_time")
    head = (f"Your current slot is {current}. " if current
            else "You don't have a counselling slot booked yet. ")
    return ChatResponse(
        reply=prefix + head + f"Here are the next available slots with {counsellor['name']}. "
              "Tap one to book it, or tell me the day/time you prefer.",
        quick_replies=[o["label"] for o in options],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks):
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

        # Auto-allocate an admission counsellor (least-loaded). If this fails for
        # any reason the registration + credentials email still go through as before.
        counsellor = None
        try:
            counsellor = counsellors.allocate_counsellor(
                student_name=data["name"], student_email=data["email"], student_mobile=data["mobile"],
                course=data["course"], source="nest", reference=username,
                extra={"percentage": data["percentage"], "category": data["category"],
                       "state": data["state"], "lead_label": label, "lead_score": score},
            )
            if counsellor:
                background_tasks.add_task(briefs.generate_and_store_brief, counsellor["assignment_id"])
        except Exception as e:
            print(f"[COUNSELLOR ERROR] Allocation failed for NEST registration: {e}")

        test_url_for_email = f"{NEST_TEST_PORTAL_URL}?user={username}"
        email_service.send_nest_credentials(data["email"], data["name"], username, password,
                                            test_url_for_email, counsellor=counsellor)
        nest["active"] = False
        nest["offered"] = False
        nest["last_username"] = username

        reply = (
            f"You're registered for AGC NEST! Your login details have also been emailed to {data['email']}.\n\n"
            f"Username: {username}\nPassword: {password}\n\n"
        )
        if counsellor:
            reply += (f"Your admission counsellor is {counsellor['name']} "
                      f"(Mobile: {counsellor['mobile']}, Email: {counsellor['email']}).\n\n")
        reply += "Click below when you're ready to take the test."
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

    # ---------------- Counselling slot booking / rescheduling ----------------
    slot_state = session.setdefault("slot", {"options": []})
    lowered = text.lower().strip()
    if slot_state["options"] and lowered in SLOT_CANCEL_WORDS:
        slot_state["options"] = []
        return ChatResponse(reply="No problem - your slot stays as it is. Anything else I can help with?")
    chosen = next((o for o in slot_state["options"] if o["label"].lower() == lowered), None)
    strong = any(k in lowered for k in SLOT_STRONG_TRIGGERS)
    weak = any(k in lowered for k in SLOT_WEAK_TRIGGERS)
    follow_up = bool(slot_state["options"]) and counsellors.has_slot_preference(text)
    if chosen or strong or weak or follow_up:
        slot_response = _handle_slot_request(session, text, chosen, strong or follow_up)
        if slot_response is not None:
            return slot_response

    # ---------------- Normal agentic conversation ----------------

    # If the student EXPLICITLY asked to register (not just asking about NEST
    # in general), skip the Yes/No offer entirely and start registration now.
    if not nest["active"] and any(k in text.lower() for k in DIRECT_REGISTRATION_KEYWORDS):
        nest["active"] = True
        nest["offered"] = False
        nest["step"] = 0
        nest["data"] = {}
        return ChatResponse(reply=NEST_PROMPTS[NEST_FIELDS[0]])

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

    try:
        briefs.save_chat_turn(req.session_id, text, reply)
    except Exception as e:
        print(f"[CHAT STORE ERROR] Could not save chat turn: {e}")

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
    counsellor_name: str = ""
    counsellor_mobile: str = ""
    counsellor_email: str = ""


@app.post("/apply", response_model=ApplicationResponse)
def apply(req: ApplicationRequest, background_tasks: BackgroundTasks):
    # Smart-allocate a counsellor and book THEIR earliest free slot. Never blocks the
    # application: if anything fails we fall back to the old round-robin slot.
    counsellor = None
    meeting_time = None
    try:
        counsellor = counsellors.allocate_counsellor(
            student_name=req.name, student_email=req.email, student_mobile=req.mobile,
            course=req.program, source="application",
            reference=f"{req.mobile}|{req.program}",
            extra={"category": req.category or None, "state": req.state or None},
        )
        if counsellor:
            meeting_time = counsellors.book_first_free_slot(counsellor["id"], counsellor["assignment_id"])
    except Exception as e:
        print(f"[COUNSELLOR ERROR] Allocation failed for application: {e}")
    if not meeting_time:
        meeting_time = pick_next_meeting_slot()

    try:
        db.save_application(req.name, req.mobile, req.email, req.program, req.state, req.category, meeting_time)
        print(f"[DB] Application saved: {req.name} - {req.program} - Meeting: {meeting_time}")
    except Exception as e:
        print(f"[DB ERROR] Failed to save application: {e}")
        if counsellor:
            try:
                counsellors.release_assignment(counsellor["assignment_id"])   # don't leave an orphan allocation
            except Exception as release_error:
                print(f"[COUNSELLOR ERROR] Could not release assignment: {release_error}")
        return ApplicationResponse(
            success=False,
            message="We couldn't save your application right now. Please try again or call +91 8872009950.",
        )

    if counsellor:
        background_tasks.add_task(briefs.generate_and_store_brief, counsellor["assignment_id"])

    if req.email:
        email_service.send_application_confirmation(req.email, req.name, req.program, meeting_time,
                                                    counsellor=counsellor)

    return ApplicationResponse(
        success=True,
        message="Application submitted successfully!",
        meeting_time=meeting_time,
        counsellor_name=counsellor["name"] if counsellor else "",
        counsellor_mobile=counsellor["mobile"] if counsellor else "",
        counsellor_email=counsellor["email"] if counsellor else "",
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


# ---------------- Admin: manual "admission reporting" email ----------------
# Sent from the NEST Registrations tab of admin.html ("Send Email" button).
# Admin picks the reporting date/time from a calendar/time picker in the
# modal and fills in counsellor + admission process details; this does NOT
# touch the automatic NEST-credentials email sent in the /chat flow above.

class AdmissionEmailRequest(BaseModel):
    username: str = ""          # links back to the NEST registration (optional)
    email: str
    name: str
    course: str
    counsellor_name: str
    counsellor_mobile: str
    counsellor_email: str
    reporting_date: str         # "YYYY-MM-DD", from <input type="date">
    reporting_time: str         # "HH:MM" 24h, from <input type="time">
    admission_process: str
    documents_required: str


class AdmissionEmailResponse(BaseModel):
    success: bool
    message: str


def _pretty_date_time(reporting_date: str, reporting_time: str):
    """Turns '2026-09-24' / '11:00' into '24 September 2026' / '11:00 AM' for the email body."""
    try:
        pretty_date = datetime.strptime(reporting_date, "%Y-%m-%d").strftime("%d %B %Y")
    except ValueError:
        pretty_date = reporting_date
    try:
        pretty_time = datetime.strptime(reporting_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
    except ValueError:
        pretty_time = reporting_time
    return pretty_date, pretty_time


@app.post("/admin/send-admission-email", response_model=AdmissionEmailResponse)
def send_admission_email(req: AdmissionEmailRequest, x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    pretty_date, pretty_time = _pretty_date_time(req.reporting_date, req.reporting_time)

    sent = email_service.send_admission_reporting_email(
        to_email=req.email,
        name=req.name,
        course=req.course,
        counsellor_name=req.counsellor_name,
        counsellor_mobile=req.counsellor_mobile,
        counsellor_email=req.counsellor_email,
        reporting_date=pretty_date,
        reporting_time=pretty_time,
        admission_process=req.admission_process,
        documents_required=req.documents_required,
    )

    try:
        db.save_admission_reporting_email(
            username=req.username,
            student_name=req.name,
            student_email=req.email,
            course=req.course,
            counsellor_name=req.counsellor_name,
            counsellor_mobile=req.counsellor_mobile,
            counsellor_email=req.counsellor_email,
            reporting_date=req.reporting_date,
            reporting_time=req.reporting_time,
            admission_process=req.admission_process,
            documents_required=req.documents_required,
        )
    except Exception as e:
        print(f"[DB ERROR] Failed to save admission reporting email record: {e}")

    if not sent:
        return AdmissionEmailResponse(
            success=False,
            message="Email failed to send. Check RESEND_API_KEY / email service configuration.",
        )
    return AdmissionEmailResponse(success=True, message=f"Email sent to {req.email}")


@app.get("/admin/admission-emails")
def admin_admission_emails(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"admission_emails": db.get_all_admission_reporting_emails()}


# ---------------- Counsellor allocation: admin + counsellor dashboard ----------------

@app.get("/admin/assignments")
def admin_assignments(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"assignments": counsellors.get_all_assignments()}


@app.get("/admin/counsellors")
def admin_counsellors(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"counsellors": counsellors.get_counsellor_load()}


def _require_counsellor(access_key: str):
    counsellor = counsellors.get_counsellor_by_key(access_key)
    if not counsellor:
        raise HTTPException(status_code=401, detail="Invalid counsellor key")
    return counsellor


@app.get("/counsellor/me")
def counsellor_me(x_counsellor_key: str = Header(default="")):
    """Polled by counsellor.html every few seconds: who am I + my assigned students."""
    counsellor = _require_counsellor(x_counsellor_key)
    return {
        "counsellor": {"name": counsellor["name"], "mobile": counsellor["mobile"], "email": counsellor["email"]},
        "assignments": counsellors.get_assignments_for_counsellor(counsellor["id"]),
        "statuses": counsellors.VALID_STATUSES,
    }


class StatusUpdate(BaseModel):
    status: str


@app.post("/counsellor/assignments/{assignment_id}/status")
def counsellor_update_status(assignment_id: int, body: StatusUpdate,
                             x_counsellor_key: str = Header(default="")):
    counsellor = _require_counsellor(x_counsellor_key)
    try:
        ok = counsellors.update_assignment_status(assignment_id, counsellor["id"], body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid status")
    if not ok:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return {"success": True}


@app.post("/counsellor/assignments/{assignment_id}/brief")
def counsellor_refresh_brief(assignment_id: int, x_counsellor_key: str = Header(default="")):
    """Counsellor clicks 'Refresh brief' - rebuilds it from the latest chat history."""
    counsellor = _require_counsellor(x_counsellor_key)
    assignment = counsellors.get_assignment(assignment_id)
    if not assignment or assignment["counsellor_id"] != counsellor["id"]:
        raise HTTPException(status_code=404, detail="Assignment not found")
    brief = briefs.generate_and_store_brief(assignment_id)
    if brief is None:
        raise HTTPException(status_code=500, detail="Could not generate brief")
    return {"brief": brief}


class AskRequest(BaseModel):
    question: str


@app.post("/admin/ask")
def admin_ask(body: AskRequest, x_admin_key: str = Header(default="")):
    """Natural-language question -> fixed read-only tools -> answer written by Python."""
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    question = body.question.strip()[:300]
    if not question:
        raise HTTPException(status_code=400, detail="Please type a question")
    return admin_agent.answer_question(question)