"""
db.py
Supabase (hosted Postgres) storage for chat leads (name, mobile, email)
captured at the start of a conversation.

Setup required before this works:
1. Create a free project at https://supabase.com
2. In the Supabase SQL Editor, run:

   create table leads (
     id bigint generated always as identity primary key,
     session_id text unique not null,
     name text,
     mobile text,
     email text,
     created_at timestamptz default now()
   );

3. Get your Project URL and anon/service key from Project Settings > API
4. Put them in a .env file in the project root:

   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_KEY=your-anon-or-service-key
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Explicitly point to the .env file in the project root, so this works
# regardless of which folder you run 'uvicorn' from.
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_KEY. Add them to a .env file "
        "in the project root - see the setup instructions at the top of db.py."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    # No-op: the 'leads' table is created once via the Supabase SQL Editor
    # (see setup instructions above), not from application code.
    pass


def log_course_interest(session_id: str, course_name: str):
    """
    Records that a student showed interest in (asked about) a specific course.
    Uses upsert with the unique(session_id, course_name) constraint so the same
    course is never logged twice for the same session.
    """
    supabase.table("course_interest").upsert({
        "session_id": session_id,
        "course_name": course_name,
    }, on_conflict="session_id,course_name").execute()


def was_course_already_logged(session_id: str, course_name: str) -> bool:
    result = (
        supabase.table("course_interest")
        .select("id")
        .eq("session_id", session_id)
        .eq("course_name", course_name)
        .execute()
    )
    return len(result.data) > 0


def find_existing_lead(mobile: str, email: str):
    """
    Checks if a lead with this mobile or email already exists (from a different
    session). Returns the matching row if found, else None.
    """
    result = (
        supabase.table("leads")
        .select("*")
        .or_(f"mobile.eq.{mobile},email.eq.{email}")
        .execute()
    )
    return result.data[0] if result.data else None


def save_lead(session_id: str, name: str, mobile: str, email: str):
    supabase.table("leads").upsert({
        "session_id": session_id,
        "name": name,
        "mobile": mobile,
        "email": email,
    }, on_conflict="session_id").execute()


def get_all_leads():
    result = (
        supabase.table("leads")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data