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

load_dotenv()

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