"""
followups.py
PROACTIVE follow-up agent.

  * A scheduler (or the admin's "Run check now" button) finds students whose counsellor
    has NOT contacted them for FOLLOWUP_AFTER_HOURS (default 48), Hot leads first.
  * For each one (max MAX_DRAFTS_PER_RUN per run, to keep the laptop light) the agent
    DECIDES with one tool call: draft_email to the student, remind_counsellor, or skip.
  * Python validates the text (length, no promises/discounts, no invented numbers, never
    mentions lead labels) and stores it as a PENDING draft.
  * NOTHING is sent until the admin clicks Approve - the model never emails anyone itself.
  * If the model is unavailable, a plain template draft is created instead.
"""

import os
import re
from datetime import datetime, timezone

import agent_runtime as rt
from agent_runtime import Tool, ToolError
import counsellors
import briefs
import email_service
from db import supabase

STALE_AFTER_HOURS = int(os.environ.get("FOLLOWUP_AFTER_HOURS", "48"))
MAX_DRAFTS_PER_RUN = 3

BANNED = re.compile(
    r"guarantee|assured|100\s*%|discount|free admission|waiver|scholarship of|\bRs\.?\s*\d|₹|"
    r"hot lead|warm lead|cold lead|lead score|lead label", re.I)

SYSTEM = (
    "You handle follow-ups for college admission students whose counsellor has not contacted them yet. "
    "Decide with exactly ONE tool call: draft_email (a short, warm email to the student, written as the "
    "counsellor), remind_counsellor (an internal note to the counsellor), or skip (no follow-up needed). "
    "Rules: never promise admission, scholarships, discounts or fees; use only the facts given; "
    "email body 40-90 words; sign off with the counsellor's name."
)
EMAIL_PARAMS = {"type": "object", "properties": {
    "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["subject", "body"]}
REMIND_PARAMS = {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}
SKIP_PARAMS = {"type": "object", "properties": {"reason": {"type": "string"}}, "required": []}


# ---------------------------------------------------------------- finding stale leads

def find_stale_leads(hours: int = None, limit: int = 20):
    hours = STALE_AFTER_HOURS if hours is None else hours
    now = datetime.now(timezone.utc)
    already = set()
    try:
        for d in supabase.table("followup_drafts").select("*").execute().data:
            if d.get("status") in ("pending", "sent", "dismissed"):
                already.add(d["assignment_id"])
    except Exception:
        pass                                    # table missing -> treat as none
    stale = []
    for r in counsellors.get_enriched_assignments():
        if r["status"] != "New" or r["id"] in already or r["lead_label"] not in ("Hot", "Warm"):
            continue
        ts = counsellors.parse_ts(r.get("created_at"))
        if ts is None:
            continue
        age_h = (now - ts).total_seconds() / 3600
        if age_h >= hours:
            r["age_hours"] = age_h
            stale.append(r)
    stale.sort(key=lambda r: (0 if r["lead_label"] == "Hot" else 1, -r["age_hours"]))
    return stale[:limit]


# ---------------------------------------------------------------- the agent

def _facts_for(row):
    topics = []
    try:
        messages, _ = briefs.get_chat_for_student(row.get("student_mobile"), row.get("student_email"))
        topics = briefs.detect_topics([m["content"] for m in messages if m["role"] == "user"])
    except Exception:
        pass
    days = max(1, round(row["age_hours"] / 24))
    first = (row.get("student_name") or "there").split()[0]
    facts = [
        f"Student first name: {first}",
        f"Course: {row.get('course')}",
        f"Registered about {days} day(s) ago and has not been contacted yet",
        f"Counsellor: {row.get('counsellor_name')} (mobile {row.get('counsellor_mobile')}, email {row.get('counsellor_email')})",
        f"Counselling slot: {row.get('meeting_time') or 'not booked yet'}",
        f"Topics the student asked about: {', '.join(topics) if topics else 'none recorded'}",
    ]
    return "\n".join(facts), first


def _template_email(row, first):
    subject = f"Following up on your {row.get('course')} enquiry - AGC Admissions"
    body = (f"Dear {first},\n\nThank you for your interest in {row.get('course')} at Amritsar Group of Colleges. "
            f"I am {row.get('counsellor_name')}, your admission counsellor, and I would be happy to answer your "
            f"questions and guide you through the admission process.\n\nYou can reach me on "
            f"{row.get('counsellor_mobile')} or reply to this email.\n\nRegards,\n{row.get('counsellor_name')}\n"
            "Amritsar Group of Colleges")
    return subject, body


def _store(row, kind, to_email, subject, body, source):
    saved = supabase.table("followup_drafts").insert({
        "assignment_id": row["id"], "kind": kind, "to_email": to_email,
        "subject": subject, "body": body, "status": "pending", "source": source,
    }).execute().data[0]
    return saved


def draft_for_lead(row):
    """Runs the agent for one stale lead. Returns the stored draft row, or None if skipped."""
    facts, first = _facts_for(row)
    allowed_numbers = rt.numbers_in(facts)

    def draft_email(args):
        subject = re.sub(r"\s+", " ", str(args.get("subject") or "")).strip()
        body = str(args.get("body") or "").strip()
        if not 5 <= len(subject) <= 90:
            raise ToolError("subject must be 5-90 characters")
        if not 80 <= len(body) <= 900:
            raise ToolError("body must be about 40-90 words (80-900 characters)")
        if BANNED.search(subject + " " + body):
            raise ToolError("Do not mention promises, discounts, waivers, fees, or lead labels. Rewrite without them.")
        if not rt.numbers_in(subject + " " + body) <= allowed_numbers:
            raise ToolError("Use only numbers that appear in the facts. Rewrite without other numbers.")
        return {"kind": "email", "subject": subject, "body": body}

    def remind(args):
        note = re.sub(r"\s+", " ", str(args.get("note") or "")).strip()
        if not 20 <= len(note) <= 300:
            raise ToolError("note must be 20-300 characters")
        if not rt.numbers_in(note) <= allowed_numbers:
            raise ToolError("Use only numbers that appear in the facts.")
        return {"kind": "reminder", "note": note}

    def skip(args):
        return {"kind": "skip", "reason": str(args.get("reason") or "")[:120]}

    def fallback():
        subject, body = _template_email(row, first)
        return {"kind": "email", "subject": subject, "body": body, "template": True}

    result, meta = rt.run(
        "followup", SYSTEM, facts,
        [Tool("draft_email", "Draft an email to the student, written as their counsellor.", EMAIL_PARAMS, draft_email, final=True),
         Tool("remind_counsellor", "Write an internal reminder note for the counsellor.", REMIND_PARAMS, remind, final=True),
         Tool("skip", "No follow-up is needed right now.", SKIP_PARAMS, skip, final=True)],
        fallback=fallback,
        describe=lambda r: f"{r['kind']}" + (" (template)" if r.get("template") else ""),
        max_steps=3, timeout=40, deadline_s=80, lock_wait=60, max_tokens=220,
    )
    if not result or result["kind"] == "skip":
        return None
    source = "template" if result.get("template") else "agent"
    if result["kind"] == "email":
        return _store(row, "email", row.get("student_email"), result["subject"], result["body"], source)
    return _store(row, "reminder", row.get("counsellor_email"),
                  f"Reminder: follow up with {row.get('student_name')}", result["note"], source)


def run_followup_check(max_drafts: int = MAX_DRAFTS_PER_RUN):
    stale = find_stale_leads()
    made = 0
    for row in stale[:max_drafts]:
        try:
            if draft_for_lead(row):
                made += 1
        except Exception as e:
            print(f"[FOLLOWUP ERROR] {row.get('student_name')}: {e}")
    print(f"[FOLLOWUP] stale leads: {len(stale)}, drafts created: {made}")
    return {"stale_found": len(stale), "drafts_created": made}


def draft_for_assignment_ids(ids):
    """Used by the admin agent ('follow up those students'). Runs in a background task."""
    wanted = {int(i) for i in ids}
    now = datetime.now(timezone.utc)
    for r in counsellors.get_enriched_assignments():
        if r["id"] in wanted:
            ts = counsellors.parse_ts(r.get("created_at"))
            r["age_hours"] = (now - ts).total_seconds() / 3600 if ts else 0
            try:
                draft_for_lead(r)
            except Exception as e:
                print(f"[FOLLOWUP ERROR] {r.get('student_name')}: {e}")


# ---------------------------------------------------------------- admin actions

def list_drafts(limit: int = 30):
    drafts = (supabase.table("followup_drafts").select("*").order("created_at", desc=True).execute().data)[:limit]
    people = {r["id"]: r for r in counsellors.get_all_assignments()}
    for d in drafts:
        a = people.get(d["assignment_id"], {})
        d["student_name"], d["course"] = a.get("student_name", ""), a.get("course", "")
        d["counsellor_name"] = a.get("counsellor_name", "")
    drafts.sort(key=lambda d: 0 if d["status"] == "pending" else 1)
    return drafts


def _get_draft(draft_id):
    rows = supabase.table("followup_drafts").select("*").eq("id", draft_id).execute().data
    return rows[0] if rows else None


def approve_draft(draft_id: int):
    draft = _get_draft(draft_id)
    if not draft or draft["status"] != "pending":
        return {"ok": False, "message": "Draft not found or already handled"}
    sent = email_service.send_followup_email(draft["to_email"], draft["subject"], draft["body"])
    supabase.table("followup_drafts").update({
        "status": "sent" if sent else "failed",
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", draft_id).execute()
    return {"ok": bool(sent), "message": "Email sent" if sent else "Email failed - check the email service"}


def dismiss_draft(draft_id: int):
    draft = _get_draft(draft_id)
    if not draft or draft["status"] != "pending":
        return {"ok": False, "message": "Draft not found or already handled"}
    supabase.table("followup_drafts").update({
        "status": "dismissed", "decided_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", draft_id).execute()
    return {"ok": True, "message": "Dismissed"}