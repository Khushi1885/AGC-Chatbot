# AGC-Chatbot

An agentic AI admission assistant for Amritsar Group of Colleges (AGC), built with Ollama (Qwen3) for local LLM tool-calling, FastAPI backend, and Supabase for lead storage.

## Features
- Agentic chatbot (tool-calling, not RAG) that answers questions about courses, fees, eligibility, admission procedure, counselling schedule, hostel, scholarships, anti-ragging policy, and grievance redressal
- Lead-capture flow: collects student name, mobile, and email before the chat begins, stored in Supabase
- Admin dashboard to view captured leads
- Reads real data from structured JSON files and CSV files

## Tech Stack
- **LLM**: Ollama (Qwen3) - runs locally, no API costs
- **Backend**: FastAPI (Python)
- **Database**: Supabase (Postgres)
- **Frontend**: Vanilla HTML/CSS/JS chat widget + admin dashboard

## Project Structure
```
AGC-Chatbot/
├── data/
│   ├── raw/csvs/          # Real CSV data (courses, fees, admission summary)
│   └── structured/        # JSON data (dates, FAQ, contacts, policies, etc.)
├── tools/                 # Agent tool functions
├── agent/                 # Core agentic loop (Ollama tool-calling)
├── backend/                # FastAPI server + Supabase integration
├── frontend/               # Chat widget + admin dashboard
├── scripts/                 # Data conversion utilities (PDF/CSV to JSON)
└── requirements.txt
```

## Setup

1. **Install Ollama** and pull a model:
   ```
   ollama pull qwen3:8b
   ```

2. **Create a Python virtual environment**:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Set up Supabase**:
   - Create a project at [supabase.com](https://supabase.com)
   - Run this in the SQL Editor:
     ```sql
     create table leads (
       id bigint generated always as identity primary key,
       session_id text unique not null,
       name text,
       mobile text,
       email text,
       created_at timestamptz default now()
     );
     ```
   - Copy `.env.example` to `.env` and fill in your `SUPABASE_URL` and `SUPABASE_KEY`

4. **Run the backend**:
   ```
   cd backend
   uvicorn main:app --reload --port 8000
   ```

5. **Open the frontend**:
   - Open `frontend/index.html` in a browser for the chat widget
   - Open `frontend/admin.html` for the leads dashboard (default admin key: change `ADMIN_KEY` in `backend/main.py`)

## License
Built as a college project.