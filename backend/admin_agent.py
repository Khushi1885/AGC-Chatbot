"""
admin_agent.py
Lets the admin ask questions in plain English/Hinglish, e.g.
    "Hot leads from this week that haven't been contacted yet"
    "Rahul ke kitne students pending hain"
    "workload per counsellor"

HOW IT WORKS (and why it is safe)
  * The model never sees or writes SQL. It can only choose between 3 fixed,
    READ-ONLY tools: find_students, counsellor_workload, lead_summary.
  * A rule-based parser extracts the filters it is sure about (Hot/Warm/Cold,
    "baat nahi hui" -> New, "is hafte" -> 7 days, counsellor names...). These
    ALWAYS override whatever arguments the model produced, because a 1.7B
    model is unreliable at turning "is hafte" into days=7.
  * Every argument is validated against allowed values / numeric ranges.
  * The answer text is written by Python from the tool's result - the model
    cannot invent a student or a number.
  * If Ollama is down or picks nothing, the rules route the question, so the
    admin always gets an answer.
"""

import re
import requests
from datetime import datetime, timedelta, timezone
from collections import Counter

from db import supabase
import counsellors

LABELS = ["Hot", "Warm", "Cold", "Unscored"]
STATUSES = counsellors.VALID_STATUSES
SOURCES = ["nest", "application"]
MAX_ROWS = 15

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_students",
        "description": ("List students assigned to counsellors, with optional filters. Use for questions "
                        "like 'hot leads not contacted yet', 'students of Rahul', 'BCA registrations this week'."),
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "enum": LABELS, "description": "Lead temperature from the ML model"},
            "status": {"type": "string", "enum": STATUSES, "description": "New = not contacted yet"},
            "source": {"type": "string", "enum": SOURCES},
            "days": {"type": "integer", "description": "Only students assigned in the last N days"},
            "course_contains": {"type": "string"},
            "counsellor_name": {"type": "string"},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "counsellor_workload",
        "description": "Show how many students each counsellor has, split by status.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "lead_summary",
        "description": "Overall counts by lead label, status and source, optionally for the last N days.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer"}}, "required": []}}},
]


# ---------------------------------------------------------------- data access (read-only)

def _parse_ts(value):
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _load_rows():
    rows = counsellors.get_all_assignments()
    regs = {}
    try:   # older assignments may not carry the ML label - borrow it from the NEST registration
        for r in supabase.table("agc_nest_registrations").select("*").execute().data:
            regs[r.get("username")] = r
    except Exception:
        pass
    for r in rows:
        if not r.get("lead_label") and r.get("source") == "nest":
            reg = regs.get(r.get("reference"))
            if reg:
                r["lead_label"], r["lead_score"] = reg.get("lead_label"), reg.get("lead_score")
        r["lead_label"] = (r.get("lead_label") or "Unscored").title()
        r["status"] = r.get("status") or "New"
    return rows


def _within_days(row, days):
    if not days:
        return True
    ts = _parse_ts(row.get("created_at"))
    return ts is None or ts >= datetime.now(timezone.utc) - timedelta(days=days)


# ---------------------------------------------------------------- tools

def find_students(label=None, status=None, source=None, days=None,
                  course_contains=None, counsellor_name=None):
    rows = _load_rows()
    out = []
    for r in rows:
        if label and r["lead_label"].lower() != label.lower():
            continue
        if status and r["status"] != status:
            continue
        if source and r.get("source") != source:
            continue
        if course_contains and course_contains.lower() not in (r.get("course") or "").lower():
            continue
        if counsellor_name and counsellor_name.lower() not in (r.get("counsellor_name") or "").lower():
            continue
        if not _within_days(r, days):
            continue
        out.append(r)

    parts = []
    if label: parts.append(label)
    if status: parts.append("not contacted yet" if status == "New" else status)
    if source: parts.append("AGC NEST" if source == "nest" else "admission inquiries")
    if course_contains: parts.append(f"course contains '{course_contains}'")
    if counsellor_name: parts.append(f"counsellor {counsellor_name}")
    if days: parts.append(f"last {days} days")
    applied = ", ".join(parts) or "none (showing everyone)"

    if not out:
        text = f"No students match these filters: {applied}."
    else:
        shown = out[:MAX_ROWS]
        lines = [f"{len(out)} student(s) match - filters: {applied}."]
        for r in shown:
            lines.append(f"- {r.get('student_name')} | {r.get('course')} | {r['lead_label']} | "
                         f"{r['status']} | {r.get('counsellor_name')}")
        if len(out) > MAX_ROWS:
            lines.append(f"...and {len(out) - MAX_ROWS} more.")
        text = "\n".join(lines)
    return {"display_text": text, "rows": out[:MAX_ROWS], "total": len(out)}


def counsellor_workload():
    people = counsellors.get_counsellor_load()
    rows = _load_rows()
    by = {}
    for r in rows:
        by.setdefault(r["counsellor_id"], Counter())[r["status"]] += 1
    lines = ["Counsellor workload (students by status):"]
    for c in people:
        cnt = by.get(c["id"], Counter())
        lines.append(f"- {c['name']}: {c['assigned']} total, {c['open']} open "
                     f"(New {cnt['New']}, Contacted {cnt['Contacted']}, "
                     f"Reported {cnt['Reported']}, Admitted {cnt['Admitted']})")
    return {"display_text": "\n".join(lines), "rows": [], "total": len(people)}


def lead_summary(days=None):
    rows = [r for r in _load_rows() if _within_days(r, days)]
    labels, statuses, sources = (Counter(r["lead_label"] for r in rows),
                                 Counter(r["status"] for r in rows),
                                 Counter(r.get("source") for r in rows))
    period = f"last {days} days" if days else "all time"
    text = (f"Summary ({period}): {len(rows)} students assigned.\n"
            f"- By lead: " + ", ".join(f"{k} {v}" for k, v in labels.most_common()) + "\n"
            f"- By status: " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()) + "\n"
            f"- By source: " + ", ".join(f"{('AGC NEST' if k == 'nest' else 'Inquiry')} {v}"
                                          for k, v in sources.most_common()))
    return {"display_text": text, "rows": [], "total": len(rows)}


TOOLS = {"find_students": find_students, "counsellor_workload": counsellor_workload,
         "lead_summary": lead_summary}


# ---------------------------------------------------------------- rule-based understanding

def parse_filters(question: str, counsellor_names=None):
    """Filters the rules are confident about. These override model-produced arguments."""
    q = (question or "").lower()
    f = {}

    for lab in ("hot", "warm", "cold"):
        if re.search(rf"\b{lab}\b", q):
            f["label"] = lab.capitalize()

    if re.search(r"not contacted|baat nahi|baat nahin|baat nhi|contact nahi|contact nhi|"
                 r"untouched|pending|follow.?up pending|\bnew\b|abhi tak nahi", q):
        f["status"] = "New"
    elif re.search(r"\bcontacted\b|baat ho (gayi|chuki|gai)", q):
        f["status"] = "Contacted"
    elif re.search(r"\breported\b", q):
        f["status"] = "Reported"
    elif re.search(r"\badmitted\b|admission le liya|\bjoined\b", q):
        f["status"] = "Admitted"

    m = re.search(r"(?:last|past|pichle|pichhle)\s+(\d{1,3})\s*(?:days?|din)", q)
    if m:
        f["days"] = int(m.group(1))
    elif re.search(r"\btoday\b|\baaj\b", q):
        f["days"] = 1
    elif re.search(r"this week|is hafte|is week|is saptah|\bhafte\b", q):
        f["days"] = 7
    elif re.search(r"this month|is mahine|is month", q):
        f["days"] = 30

    if re.search(r"\bnest\b|scholarship", q):
        f["source"] = "nest"
    elif re.search(r"application|inquiry|enquiry|\bform\b", q):
        f["source"] = "application"

    for name in (counsellor_names or []):
        first = name.split()[0].lower()
        if re.search(rf"\b{re.escape(first)}\b", q) or name.lower() in q:
            f["counsellor_name"] = name
            break

    for kw in ("b.tech", "btech", "bca", "mca", "mba", "bba", "b.com", "pharmacy", "bhmct", "agriculture"):
        if re.search(rf"\b{re.escape(kw)}\b", q):
            f["course_contains"] = "b.tech" if kw in ("b.tech", "btech") else kw
            break
    return f


def detect_intent(question: str):
    q = (question or "").lower()
    if re.search(r"workload|work load|per counsellor|counsellor.?wise|kaun sa counsellor|"
                 r"who has (the )?(most|least)|sabse (zyada|kam)", q):
        return "workload"
    if re.search(r"summary|overview|stats|statistics|breakdown|total", q):
        return "summary"
    return None


def sanitize_args(args: dict) -> dict:
    """Validate model-produced arguments; drop anything unexpected."""
    clean = {}
    if not isinstance(args, dict):
        return clean
    if isinstance(args.get("label"), str) and args["label"].capitalize() in LABELS:
        clean["label"] = args["label"].capitalize()
    if args.get("status") in STATUSES:
        clean["status"] = args["status"]
    if args.get("source") in SOURCES:
        clean["source"] = args["source"]
    try:
        if args.get("days") is not None:
            d = int(args["days"])
            if 1 <= d <= 365:
                clean["days"] = d
    except (TypeError, ValueError):
        pass
    for key in ("course_contains", "counsellor_name"):
        if isinstance(args.get(key), str) and args[key].strip():
            clean[key] = re.sub(r"[^\w\s.\-]", "", args[key])[:40].strip()
    return clean


# ---------------------------------------------------------------- LLM tool routing

def llm_route(question: str):
    """Ask the local model which tool fits. Returns (tool_name, args) or (None, {})."""
    try:
        from agent import OLLAMA_URL, MODEL_NAME
    except Exception:
        OLLAMA_URL, MODEL_NAME = "http://localhost:11434/api/chat", "qwen3:1.7b"
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "stream": False, "think": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": (
                    "You help a college admission admin query lead data. Always answer by calling exactly one "
                    "tool. Lead label comes from an ML model (Hot/Warm/Cold). Status 'New' means the counsellor "
                    "has not contacted the student yet.")},
                {"role": "user", "content": question},
            ],
            "tools": TOOL_SCHEMAS,
        }, timeout=25)
        calls = resp.json().get("message", {}).get("tool_calls") or []
        if calls:
            fn = calls[0]["function"]
            if fn.get("name") in TOOLS:
                return fn["name"], fn.get("arguments") or {}
    except Exception as e:
        print(f"[ADMIN AGENT] LLM routing unavailable, using rules: {e}")
    return None, {}


def answer_question(question: str):
    names = []
    try:
        names = [c["name"] for c in counsellors.get_counsellor_load()]
    except Exception:
        pass

    hints = parse_filters(question, names)
    intent = detect_intent(question)
    llm_tool, llm_args = llm_route(question)

    # Rules win when they are confident; the LLM decides only for open-ended phrasing.
    if intent == "workload":
        tool = "counsellor_workload"
    elif intent == "summary" and not {k for k in hints if k != "days"}:
        tool = "lead_summary"
    elif hints:
        tool = "find_students"
    else:
        tool = llm_tool or "find_students"

    args = {}
    if tool == "find_students":
        args = {**sanitize_args(llm_args), **hints}
    elif tool == "lead_summary":
        args = {k: v for k, v in {**sanitize_args(llm_args), **hints}.items() if k == "days"}

    result = TOOLS[tool](**args)
    return {
        "answer": result["display_text"],
        "rows": result["rows"],
        "total": result["total"],
        "tool": tool,
        "filters": args,
        "routed_by": "rules" if (intent or hints) else ("llm" if llm_tool else "default"),
    }