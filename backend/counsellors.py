"""
counsellors.py
Admission-counsellor allocation, slot booking and dashboard data.

SMART MATCHING (rule-based, uses the ML lead label as one input)
  1. Map the student's course to a discipline (Engineering, Management, ...).
  2. Prefer counsellors who specialise in that discipline.
  3. Among them pick the one with the fewest OPEN students (not yet Admitted);
     ties go to whoever has fewer open HOT leads (so hot leads spread out),
     then to the lowest id.
  4. Overflow rule: if the best specialist already carries MAX_LOAD_GAP more
     open students than the least-loaded counsellor overall, pick that
     least-loaded counsellor instead, so specialists never get buried.
  5. No specialist for that discipline -> least-loaded counsellor overall.

SLOT BOOKING
  Each counsellor offers SLOT_TIMES on business days (Mon-Fri). A slot is free
  unless it is in 'counsellor_bookings' or the date is in
  'counsellor_unavailable' (leave days). A unique constraint on
  (counsellor_id, slot_date, slot_time) prevents double booking.

Uses the same Supabase client as db.py, so db.py needs no changes.
Setup SQL: see migration_v2.sql.
"""

import re
from datetime import date, datetime, timedelta, timezone

from db import supabase

VALID_STATUSES = ["New", "Contacted", "Reported", "Admitted"]
CLOSED_STATUSES = {"Admitted"}

SLOT_TIMES = ["10:00 AM", "11:30 AM", "2:00 PM", "3:30 PM"]
MAX_LOAD_GAP = 5
PROFILE_FIELDS = ("percentage", "category", "state", "lead_label", "lead_score")

# Order matters: the first discipline whose keyword appears in the course name wins.
DISCIPLINE_KEYWORDS = [
    ("Hospitality", ["hotel", "hmct", "bhmct", "hcm", "hospitality", "tourism", "culinary"]),
    ("Pharmacy", ["pharm"]),
    ("Agriculture", ["agri"]),
    ("Paramedical", ["medical lab", "nutrition", "radiology", "operation theatre",
                     "paramedical", "nursing", "physiotherapy", "optometry"]),
    ("Design", ["fashion", "design"]),
    ("Computer Applications", ["bca", "mca", "computer application"]),
    ("Management", ["mba", "bba", "b.com", "m.com", "commerce", "management",
                    "business administration"]),
    ("Engineering", ["b.tech", "btech", "m.tech", "mtech", "engineering", "b.e "]),
]


# ---------------------------------------------------------------- helpers

def infer_discipline(course_name: str) -> str:
    name = (course_name or "").lower().strip()
    if re.search(r"\bfd\b", name):
        return "Design"
    for discipline, keywords in DISCIPLINE_KEYWORDS:
        if any(k in name for k in keywords):
            return discipline
    return "General"


def _norm_mobile(mobile) -> str:
    digits = re.sub(r"\D", "", mobile or "")
    return digits[-10:] if len(digits) >= 10 else digits


def _norm_email(email) -> str:
    return (email or "").strip().lower()


def _public(counsellor: dict) -> dict:
    """Counsellor dict without the private access_key."""
    return {k: v for k, v in counsellor.items() if k != "access_key"}


def _specs(counsellor: dict):
    raw = counsellor.get("specializations") or "General"
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


# ---------------------------------------------------------------- reading counsellors

def get_active_counsellors():
    rows = supabase.table("counsellors").select("*").eq("active", True).order("id").execute().data
    return [_public(r) for r in rows]


def get_counsellor(counsellor_id: int):
    rows = supabase.table("counsellors").select("*").eq("id", counsellor_id).execute().data
    return _public(rows[0]) if rows else None


def get_counsellor_by_key(access_key: str):
    if not access_key:
        return None
    rows = (supabase.table("counsellors").select("*")
            .eq("access_key", access_key).eq("active", True).execute().data)
    return _public(rows[0]) if rows else None


# ---------------------------------------------------------------- smart allocation

def _load_stats():
    """({counsellor_id: open students}, {counsellor_id: open HOT students})"""
    rows = supabase.table("counsellor_assignments").select("*").execute().data
    open_load, hot_load = {}, {}
    for r in rows:
        if (r.get("status") or "New") in CLOSED_STATUSES:
            continue
        cid = r["counsellor_id"]
        open_load[cid] = open_load.get(cid, 0) + 1
        if (r.get("lead_label") or "").lower() == "hot":
            hot_load[cid] = hot_load.get(cid, 0) + 1
    return open_load, hot_load


def pick_counsellor(counsellor_list, discipline, open_load, hot_load):
    """Pure decision function. Returns (counsellor, reason)."""
    def key(c):
        return (open_load.get(c["id"], 0), hot_load.get(c["id"], 0), c["id"])

    overall_best = min(counsellor_list, key=key)
    specialists = [c for c in counsellor_list if discipline.lower() in _specs(c)]
    if not specialists:
        return overall_best, "least-loaded (no specialist)"
    best = min(specialists, key=key)
    gap = open_load.get(best["id"], 0) - open_load.get(overall_best["id"], 0)
    if gap >= MAX_LOAD_GAP:
        return overall_best, "overflow (specialist overloaded)"
    return best, f"specialist ({discipline})"


def allocate_counsellor(student_name: str, student_email: str, student_mobile: str,
                        course: str, source: str, reference: str,
                        meeting_time: str = None, extra: dict = None):
    """
    Assigns a counsellor and returns {id, name, mobile, email, specializations,
    assignment_id, match_reason}, or None if no active counsellor exists.

    source:    'nest' or 'application'
    reference: stable id of this student action (NEST username, or
               "mobile|program" for applications). Allocating the same
               (source, reference) twice returns the original counsellor.
    extra:     optional profile fields stored with the assignment:
               percentage, category, state, lead_label, lead_score
    """
    counsellor_list = get_active_counsellors()
    if not counsellor_list:
        print("[COUNSELLOR] No active counsellors configured - skipping allocation")
        return None

    existing = (supabase.table("counsellor_assignments").select("*")
                .eq("source", source).eq("reference", reference).execute().data)
    if existing:
        for c in counsellor_list:
            if c["id"] == existing[0]["counsellor_id"]:
                return {**c, "assignment_id": existing[0]["id"], "match_reason": "existing assignment"}

    discipline = infer_discipline(course)
    open_load, hot_load = _load_stats()
    chosen, reason = pick_counsellor(counsellor_list, discipline, open_load, hot_load)

    row = {
        "counsellor_id": chosen["id"],
        "student_name": student_name,
        "student_email": _norm_email(student_email),
        "student_mobile": _norm_mobile(student_mobile),
        "course": course,
        "source": source,
        "reference": reference,
        "meeting_time": meeting_time,
        "status": "New",
    }
    full_row = dict(row)
    for k in PROFILE_FIELDS:
        if extra and extra.get(k) is not None:
            full_row[k] = extra[k]
    try:
        result = supabase.table("counsellor_assignments").insert(full_row).execute()
    except Exception as e:
        # Most likely migration_v2.sql has not been run yet (new profile columns missing).
        print(f"[COUNSELLOR WARNING] Insert with profile fields failed ({e}); retrying without them")
        result = supabase.table("counsellor_assignments").insert(row).execute()

    assignment_id = result.data[0]["id"]
    print(f"[COUNSELLOR] {student_name} ({source}, {discipline}) -> {chosen['name']} [{reason}]")
    return {**chosen, "assignment_id": assignment_id, "match_reason": reason}


def release_assignment(assignment_id: int):
    """Undo an allocation (used if saving the application fails afterwards)."""
    supabase.table("counsellor_bookings").delete().eq("assignment_id", assignment_id).execute()
    supabase.table("counsellor_assignments").delete().eq("id", assignment_id).execute()


# ---------------------------------------------------------------- slot preferences from chat text

_WEEKDAYS = {
    "monday": 0, "mon": 0, "somvar": 0, "somwar": 0,
    "tuesday": 1, "tue": 1, "tues": 1, "mangalvar": 1, "mangalwar": 1,
    "wednesday": 2, "wed": 2, "budhvar": 2, "budhwar": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "guruvar": 3, "veervar": 3, "guruwar": 3,
    "friday": 4, "fri": 4, "shukravar": 4, "shukrawar": 4,
    "saturday": 5, "sat": 5, "shanivar": 5, "shaniwar": 5,
    "sunday": 6, "sun": 6, "ravivar": 6, "raviwar": 6,
}
_NEGATION = {"nahi", "nahin", "nhi", "not", "can't", "cant", "cannot", "won't", "except",
             "busy", "unavailable", "without"}
_MORNING = {"morning", "subah", "early"}
_AFTERNOON = {"afternoon", "evening", "dopahar", "shaam"}


def parse_slot_preferences(text: str):
    """
    "Friday ko possible nahi"            -> exclude Friday
    "Tuesday afternoon"                  -> prefer Tuesday, afternoon
    "Monday nahi, Tuesday theek hai"     -> exclude Monday, prefer Tuesday
    """
    exclude, prefer, part = set(), set(), None
    clauses = re.split(r"[,;.\n]|\bbut\b|\blekin\b|\bmagar\b|\bhowever\b|\binstead\b|\bbalki\b",
                       (text or "").lower())
    for clause in clauses:
        tokens = re.findall(r"[a-z']+", clause)
        days = {_WEEKDAYS[t] for t in tokens if t in _WEEKDAYS}
        negated = any(t in _NEGATION for t in tokens)
        if days:
            (exclude if negated else prefer).update(days)
        if _MORNING & set(tokens):
            part = "morning"
        elif _AFTERNOON & set(tokens):
            part = "afternoon"
    return {"exclude": exclude, "prefer": prefer - exclude, "part": part}


def has_slot_preference(text: str) -> bool:
    p = parse_slot_preferences(text)
    return bool(p["exclude"] or p["prefer"] or p["part"])


# ---------------------------------------------------------------- availability + booking

def _business_days(start: date, count: int):
    out, d = [], start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _slot_dict(d: date, time_str: str):
    return {
        "date": d.isoformat(),
        "time": time_str,
        "label": f"{d.strftime('%a, %d %b')} - {time_str}",          # shown on the quick-reply button
        "full": f"{d.strftime('%A, %d %b %Y')} at {time_str}",        # stored / emailed
    }


def get_free_slots(counsellor_id: int, exclude_weekdays=None, prefer_weekdays=None,
                   part=None, limit: int = 4, business_days: int = 10, today: date = None):
    today = today or date.today()
    days = _business_days(today + timedelta(days=1), business_days)
    first = days[0].isoformat()

    booked = {(r["slot_date"], r["slot_time"]) for r in
              supabase.table("counsellor_bookings").select("*")
              .eq("counsellor_id", counsellor_id).gte("slot_date", first).execute().data}
    blocked = {r["unavailable_date"] for r in
               supabase.table("counsellor_unavailable").select("*")
               .eq("counsellor_id", counsellor_id).gte("unavailable_date", first).execute().data}

    exclude_weekdays = set(exclude_weekdays or [])
    prefer_weekdays = set(prefer_weekdays or [])
    per_day_cap = 4 if prefer_weekdays else 2          # spread options over several days by default

    out = []
    for d in days:
        if d.isoformat() in blocked or d.weekday() in exclude_weekdays:
            continue
        if prefer_weekdays and d.weekday() not in prefer_weekdays:
            continue
        taken_today = 0
        for t in SLOT_TIMES:
            if (d.isoformat(), t) in booked:
                continue
            if part == "morning" and "AM" not in t:
                continue
            if part == "afternoon" and "PM" not in t:
                continue
            out.append(_slot_dict(d, t))
            taken_today += 1
            if taken_today >= per_day_cap or len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def suggest_slots(counsellor_id: int, text: str, today: date = None):
    prefs = parse_slot_preferences(text)
    slots = get_free_slots(counsellor_id, exclude_weekdays=prefs["exclude"],
                           prefer_weekdays=prefs["prefer"], part=prefs["part"], today=today)
    return slots, prefs


def get_assignment(assignment_id: int):
    rows = supabase.table("counsellor_assignments").select("*").eq("id", assignment_id).execute().data
    return rows[0] if rows else None


def book_slot(assignment: dict, slot_date: str, slot_time: str):
    """
    Books (or moves) the student's counselling slot.
    Returns {"ok": True, "label": ...} or {"ok": False, "reason": "taken"|"unavailable"|"invalid"}.
    The old booking is released only after the new one is secured.
    """
    cid, aid = assignment["counsellor_id"], assignment["id"]
    try:
        d = date.fromisoformat(slot_date)
    except ValueError:
        return {"ok": False, "reason": "invalid"}
    if slot_time not in SLOT_TIMES or d.weekday() >= 5:
        return {"ok": False, "reason": "invalid"}

    blocked = (supabase.table("counsellor_unavailable").select("*")
               .eq("counsellor_id", cid).eq("unavailable_date", slot_date).execute().data)
    if blocked:
        return {"ok": False, "reason": "unavailable"}

    try:
        new_row = (supabase.table("counsellor_bookings")
                   .insert({"counsellor_id": cid, "slot_date": slot_date,
                            "slot_time": slot_time, "assignment_id": aid})
                   .execute().data[0])
    except Exception:
        return {"ok": False, "reason": "taken"}      # unique constraint: someone booked it first

    old = supabase.table("counsellor_bookings").select("*").eq("assignment_id", aid).execute().data
    for r in old:
        if r["id"] != new_row["id"]:
            supabase.table("counsellor_bookings").delete().eq("id", r["id"]).execute()

    label = _slot_dict(d, slot_time)["full"]
    supabase.table("counsellor_assignments").update({"meeting_time": label}).eq("id", aid).execute()

    # Keep the admin's Applications table in sync (best effort).
    if assignment.get("source") == "application" and "|" in (assignment.get("reference") or ""):
        mobile, program = assignment["reference"].split("|", 1)
        try:
            (supabase.table("applications").update({"meeting_time": label})
             .eq("mobile", mobile).eq("program", program).execute())
        except Exception as e:
            print(f"[COUNSELLOR] Could not sync applications.meeting_time: {e}")
    return {"ok": True, "label": label}


def book_first_free_slot(counsellor_id: int, assignment_id: int):
    """Gives a new assignment the counsellor's earliest free slot. Returns the label or None."""
    assignment = get_assignment(assignment_id)
    if not assignment:
        return None
    if assignment.get("meeting_time"):
        return assignment["meeting_time"]            # already has one - don't book a second
    slots = get_free_slots(counsellor_id, limit=1)
    if not slots:
        return None
    result = book_slot(assignment, slots[0]["date"], slots[0]["time"])
    return result["label"] if result["ok"] else None


# ---------------------------------------------------------------- dashboard / admin data

def find_assignment_for_student(mobile: str, email: str):
    """Newest assignment matching this student's mobile or email."""
    found = {}
    m, e = _norm_mobile(mobile), _norm_email(email)
    if m:
        for r in supabase.table("counsellor_assignments").select("*").eq("student_mobile", m).execute().data:
            found[r["id"]] = r
    if e:
        for r in supabase.table("counsellor_assignments").select("*").eq("student_email", e).execute().data:
            found[r["id"]] = r
    if not found:
        return None
    return max(found.values(), key=lambda r: (r.get("created_at") or "", r["id"]))


def get_assignments_for_counsellor(counsellor_id: int):
    return (supabase.table("counsellor_assignments").select("*")
            .eq("counsellor_id", counsellor_id).order("created_at", desc=True).execute().data)


def update_assignment_status(assignment_id: int, counsellor_id: int, status: str):
    """Only the counsellor who owns the assignment can change its status."""
    if status not in VALID_STATUSES:
        raise ValueError("Invalid status")
    result = (supabase.table("counsellor_assignments").update({"status": status})
              .eq("id", assignment_id).eq("counsellor_id", counsellor_id).execute())
    return len(result.data) > 0


def set_brief(assignment_id: int, brief: str):
    supabase.table("counsellor_assignments").update({
        "brief": brief,
        "brief_updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", assignment_id).execute()


def get_all_assignments():
    """Admin view: every assignment with the counsellor's details merged in."""
    rows = (supabase.table("counsellor_assignments").select("*")
            .order("created_at", desc=True).execute().data)
    people = {c["id"]: c for c in supabase.table("counsellors").select("*").execute().data}
    for r in rows:
        c = people.get(r["counsellor_id"], {})
        r["counsellor_name"] = c.get("name", "")
        r["counsellor_mobile"] = c.get("mobile", "")
        r["counsellor_email"] = c.get("email", "")
    return rows


def get_counsellor_load():
    """Admin view: each counsellor with specialization and student counts."""
    rows = supabase.table("counsellor_assignments").select("*").execute().data
    total, open_ = {}, {}
    for r in rows:
        cid = r["counsellor_id"]
        total[cid] = total.get(cid, 0) + 1
        if (r.get("status") or "New") not in CLOSED_STATUSES:
            open_[cid] = open_.get(cid, 0) + 1
    people = supabase.table("counsellors").select("*").order("id").execute().data
    out = []
    for c in people:
        c = _public(c)
        c["assigned"] = total.get(c["id"], 0)
        c["open"] = open_.get(c["id"], 0)
        out.append(c)
    return out