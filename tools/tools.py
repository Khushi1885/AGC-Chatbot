"""
tools.py
Defines the actual Python functions the agent can call, and the JSON schemas
that describe them to the LLM (Ollama tool-calling format).
"""

import json
import os
import difflib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "structured")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "csvs")


def _load_csv(filename):
    """Reads a CSV file and returns a list of dicts (one per row)."""
    import csv
    path = os.path.join(CSV_DIR, filename)
    if not os.path.exists(path):
        return {"error": f"CSV file '{filename}' not found in data/raw/csvs/"}
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows

# Common branch abbreviations students actually type, mapped to words that
# should appear in the full course name
COURSE_ALIASES = {
    "cse": ["computer science"],
    "cs": ["computer science"],
    "it": ["information technology"],
    "ece": ["electronics"],
    "eee": ["electrical"],
    "ee": ["electrical"],
    "mech": ["mechanical"],
    "civil": ["civil"],
    "mca": ["computer application"],
    "bca": ["computer application"],
    "ai": ["artificial intelligence"],
    "ml": ["machine learning"],
    "aiml": ["artificial intelligence"],
}


def _find_course(course_name: str, courses_list: list):
    """
    Matches a student-typed course name (e.g. 'B.Tech CSE', 'computer science')
    against the full course_name field in courses.json, handling common
    abbreviations and partial/substring matches in either direction.
    """
    query = course_name.lower().strip()
    # Strip common degree prefixes that add noise to matching
    for prefix in ["b.tech", "btech", "m.tech", "mtech", "b.", "m."]:
        query = query.replace(prefix, "").strip()

    # Expand known abbreviations (e.g. "cse" -> "computer science")
    search_terms = [query]
    for abbr, expansions in COURSE_ALIASES.items():
        if abbr == query or f" {abbr}" in f" {query}":
            search_terms.extend(expansions)

    for course in courses_list:
        full_name = course.get("course_name", "").lower()
        for term in search_terms:
            if term and (term in full_name or full_name in term):
                return course
    return None


def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Tool implementations ----------

def get_application_steps(category: str = None):
    data = _load("application_process.json")
    if not category:
        return data.get("general_admission_procedure")
    for proc in data.get("management_quota_procedures", []):
        if category.lower() in proc.get("category", "").lower():
            return proc
    return {"error": f"No procedure found matching category '{category}'"}


def get_counselling_schedule(course_type: str = None):
    data = _load("centralized_counselling_ikgptu.json")
    if not course_type:
        return data.get("schedules")
    schedules = data.get("schedules", {})
    for key, val in schedules.items():
        if course_type.lower() in key.lower():
            return val
    return {"error": f"No counselling schedule found matching '{course_type}'"}


def get_hostel_info():
    return _load("hostel_facilities.json")


def get_placement_stats(course_id: str = None, course_name: str = None):
    data = _load("placements.json")

    if course_name and not course_id:
        course_matches = get_course_info(course_name)
        if isinstance(course_matches, dict) and "error" in course_matches:
            return {"error": f"No course found matching '{course_name}'"}
        # placements.json is still sample data keyed by old course_id values;
        # note this for whoever fills in real placement data later.
        return {"note": "Real placement data not yet loaded for this course. Add it to data/structured/placements.json or a placements.csv.", "matched_course": course_matches[0] if isinstance(course_matches, list) else course_matches}

    if not course_id:
        return data

    matches = [p for p in data.get("placements", []) if p.get("course_id") == course_id]
    return matches if matches else {"error": f"No placement data found for course '{course_id}'"}


def search_faq(keyword: str):
    data = _load("faq.json")
    keyword = keyword.lower()
    matches = [
        f for f in data.get("faq", [])
        if keyword in f.get("question", "").lower()
        or keyword in f.get("answer", "").lower()
        or keyword in " ".join(f.get("tags", [])).lower()
    ]
    return matches if matches else {"error": f"No FAQ found matching '{keyword}'"}


def get_scholarship_info():
    data = _load("college_overview.json")
    return data.get("college_overview", {}).get("agc_nest_scholarship", {})


def get_anti_ragging_info():
    return _load("anti_ragging_policy.json")


def get_grievance_info(topic: str = None):
    data = _load("grievance_redressal.json")
    if topic and "women" in topic.lower():
        return data.get("women_grievance_redressal_committee")
    return data.get("student_grievance_redressal")


def _fuzzy_find_course(query: str, courses_list: list, name_key: str = "Program"):
    """
    Fallback fuzzy matcher: if exact substring/alias matching finds nothing,
    this finds the closest matching course name using similarity scoring.
    Handles typos, unlisted abbreviations, and slightly different phrasing
    without needing a hardcoded alias for every one of the 41 courses.
    """
    names = [c.get(name_key, "") for c in courses_list if c.get(name_key, "").strip()]
    close = difflib.get_close_matches(query, names, n=1, cutoff=0.4)
    if not close:
        return None
    best_name = close[0]
    for c in courses_list:
        if c.get(name_key, "") == best_name:
            return c
    return None


def _classify_level(course_name: str) -> str:
    """Classifies a course as UG, PG, or Doctorate based on its name."""
    lower = course_name.strip().lower()
    if lower.startswith("pharm.d"):
        return "Doctorate"
    if "bachelor" in lower:
        return "UG"
    if "master" in lower:
        return "PG"
    # Fallback: check the first letter of the abbreviation (e.g. "B.Sc.", "M.Tech.")
    first_word = lower.split()[0] if lower.split() else ""
    if first_word.startswith("b"):
        return "UG"
    if first_word.startswith("m"):
        return "PG"
    return "Other"


# Maps 12th-grade stream to the disciplines that typically require/suit that
# background - based on standard Indian education stream conventions.
STREAM_TO_DISCIPLINES = {
    "medical": ["Pharmacy", "Paramedical"],
    "pcb": ["Pharmacy", "Paramedical"],
    "non-medical": ["Engineering"],
    "nonmedical": ["Engineering"],
    "pcm": ["Engineering"],
    "commerce": ["Management"],
    "arts": ["Hotel Management", "Fashion Design", "Computer Applications", "Management"],
    "humanities": ["Hotel Management", "Fashion Design", "Computer Applications", "Management"],
}


def get_courses_by_stream(stream: str):
    """
    Given a 12th-grade stream (medical, non-medical, commerce, arts), returns
    courses from disciplines that typically suit that background. This lets
    students ask in natural terms ('medical stream') instead of needing to
    know exact course/discipline names.
    """
    stream_key = stream.strip().lower().replace(" ", "-")
    disciplines = STREAM_TO_DISCIPLINES.get(stream_key)
    if not disciplines:
        return {"error": f"Unrecognized stream '{stream}'. Try 'medical', 'non-medical', 'commerce', or 'arts'."}

    rows = _load_csv("courses.csv")
    if isinstance(rows, dict) and "error" in rows:
        return rows

    matches = [
        {"course": row.get("Program", "").strip(), "duration": row.get("Duration (from Program)", "").strip(), "discipline": row.get("Discipline", "").strip()}
        for row in rows
        if row.get("Discipline", "").strip() in disciplines and row.get("Program", "").strip()
    ]
    display_text = "\n".join(f"- {c['course']} ({c['duration']})" for c in matches)
    return {
        "stream": stream,
        "matched_disciplines": disciplines,
        "count": len(matches),
        "courses": matches,
        "display_text": display_text,
        "note": "This is a general guideline based on discipline. Exact eligibility should be confirmed with the admission office.",
    }


def get_all_courses_list():
    """
    Returns a clean, simplified list of ALL courses (just name + duration),
    stripped of extra columns and blank rows, plus a ready-to-display text
    block so the model can relay it directly instead of reformatting itself.
    """
    rows = _load_csv("courses.csv")
    if isinstance(rows, dict) and "error" in rows:
        return rows

    courses = [
        {"course": row.get("Program", "").strip(), "duration": row.get("Duration (from Program)", "").strip()}
        for row in rows
        if row.get("Program", "").strip()
    ]
    display_text = "\n".join(f"- {c['course']} ({c['duration']})" for c in courses)
    return {"count": len(courses), "courses": courses, "display_text": display_text}


def get_courses_by_level(level: str):
    """
    Returns courses filtered by level: 'UG', 'PG', or 'Doctorate'.
    Filtering is done in Python (reliable) rather than asking the model to
    separate UG/PG itself from a mixed list.
    """
    rows = _load_csv("courses.csv")
    if isinstance(rows, dict) and "error" in rows:
        return rows

    level_normalized = level.strip().upper()
    if level_normalized not in ("UG", "PG", "DOCTORATE"):
        return {"error": "level must be 'UG', 'PG', or 'Doctorate'"}

    matches = []
    for row in rows:
        program = row.get("Program", "").strip()
        if not program:
            continue
        if _classify_level(program).upper() == level_normalized:
            matches.append({"course": program, "duration": row.get("Duration (from Program)", "").strip()})

    display_text = "\n".join(f"- {c['course']} ({c['duration']})" for c in matches)
    return {"count": len(matches), "courses": matches, "display_text": display_text}


def get_program_summary():
    """
    Returns pre-computed aggregate stats (total programs, total intake, etc.)
    calculated in Python - never let the LLM do this math itself, since small
    models make arithmetic errors when summing long lists.
    """
    rows = _load_csv("courses.csv")
    if isinstance(rows, dict) and "error" in rows:
        return rows

    total_programs = len(rows)
    total_intake = 0
    by_discipline = {}

    for row in rows:
        intake_str = row.get("Intake", "").strip()
        intake = int(intake_str) if intake_str.isdigit() else 0
        total_intake += intake

        discipline = row.get("Discipline", "Unknown")
        by_discipline[discipline] = by_discipline.get(discipline, 0) + intake

    return {
        "total_programs": total_programs,
        "total_intake_across_all_programs": total_intake,
        "intake_by_discipline": by_discipline,
    }


def get_eligibility_info(discipline: str = None):
    """
    Returns general subject-stream eligibility norms by discipline (e.g. PCM for
    B.Tech). Does NOT include exact percentage cutoffs - those vary by institute
    and must be confirmed via escalate_to_admission_office.
    """
    data = _load("eligibility.json")
    if not discipline:
        return data

    query = discipline.lower().strip()
    matches = [
        e for e in data.get("eligibility_by_discipline", [])
        if query in e.get("discipline", "").lower()
    ]
    return matches if matches else {"error": f"No eligibility info found for '{discipline}'"}


def get_course_info(course_name: str = None):
    """
    Reads the real courses.csv (Discipline, Program, Duration, Intake).
    Omit course_name to list all courses; pass a name/abbreviation to find matches.
    """
    rows = _load_csv("courses.csv")
    if isinstance(rows, dict) and "error" in rows:
        return rows

    if not course_name:
        return rows

    query = course_name.lower().strip()
    for prefix in ["b.tech", "btech", "m.tech", "mtech", "b.", "m."]:
        query = query.replace(prefix, "").strip()

    search_terms = [query]
    for abbr, expansions in COURSE_ALIASES.items():
        if abbr == query or f" {abbr}" in f" {query}":
            search_terms.extend(expansions)

    matches = []
    for row in rows:
        program_field = row.get("Program", "").lower()
        for term in search_terms:
            if term and (term in program_field or program_field in term):
                matches.append(row)
                break

    # Fallback: if exact/alias matching found nothing, try fuzzy matching
    # against the full official course name list (catches typos, unlisted
    # abbreviations, or slightly different phrasing)
    if not matches:
        fuzzy_match = _fuzzy_find_course(course_name, rows, name_key="Program")
        if fuzzy_match:
            matches = [fuzzy_match]

    return matches if matches else {"error": f"No course found matching '{course_name}'"}


def get_semester_fee_structure(course_name: str):
    """
    Reads the semester-wise fee CSV (columns: Course, 1st Sem...12th Sem, Total)
    and returns the fee breakdown for a matching course, with comma-formatted
    numbers cleaned into real numbers and '—' treated as not applicable.
    """
    rows = _load_csv("fees.csv")
    if isinstance(rows, dict) and "error" in rows:
        return rows

    def clean_number(val):
        if val is None:
            return None
        val = str(val).strip()
        if val.lower() in ("—", "-", "", "n/a", "na", "not applicable", "none", "nil"):
            return None
        try:
            return int(val.replace(",", ""))
        except ValueError:
            return val

    query = course_name.lower().strip()
    for prefix in ["b.tech", "btech", "m.tech", "mtech"]:
        query = query.replace(prefix, "").strip()

    # Ambiguous query (e.g. student just said "btech fee" with no specific branch)
    # -> return the list of real branch names as options instead of guessing.
    if query in ("", "engineering", "tech"):
        branch_names = sorted(set(
            row.get("Course", "") for row in rows
            if row.get("Course", "").lower().startswith(("b.tech", "m.tech"))
        ))
        return {"error": "ambiguous", "options": branch_names}

    search_terms = [query]
    for abbr, expansions in COURSE_ALIASES.items():
        if abbr == query or f" {abbr}" in f" {query}":
            search_terms.extend(expansions)

    match = None
    for row in rows:
        course_field = row.get("Course", "").lower()
        for term in search_terms:
            if term and (term in course_field or course_field in term):
                match = row
                break
        if match:
            break

    # Fallback: fuzzy match against the official fee-list course names
    if not match:
        match = _fuzzy_find_course(course_name, rows, name_key="Course")

    if not match:
        return {"error": f"No fee data found for course matching '{course_name}'"}

    semester_fees = {}
    for key, val in match.items():
        if "Sem" in key:
            cleaned = clean_number(val)
            if cleaned is not None:
                semester_fees[key] = cleaned

    total_fee = clean_number(match.get("Total"))

    def format_fee(val):
        """Safely formats a fee value - handles both real numbers and any
        unexpected non-numeric text without crashing."""
        if isinstance(val, int):
            return f"Rs. {val:,}"
        return f"Rs. {val}"

    display_lines = [f"- {sem}: {format_fee(fee)}" for sem, fee in semester_fees.items()]
    display_lines.append(f"- Total ({len(semester_fees)} semesters): {format_fee(total_fee)}")
    display_text = "\n".join(display_lines)

    return {
        "course": match.get("Course"),
        "semester_wise_fees": semester_fees,
        "total_fee": total_fee,
        "number_of_semesters": len(semester_fees),
        "display_text": display_text,
    }


def escalate_to_admission_office(region: str = None):
    data = _load("contact_escalation.json")
    if region:
        for officer in data.get("admission_officers", []):
            if region.lower() in officer.get("region", "").lower():
                return officer
    return data.get("primary_contact")


def list_available_csv_files():
    """Lists which CSV files exist in data/raw/csvs/, so the agent knows what it can query."""
    if not os.path.exists(CSV_DIR):
        return {"error": "No CSV folder found. Put your CSV files in data/raw/csvs/"}
    files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    return {"available_csv_files": files}


def query_csv(filename: str, filter_column: str = None, filter_value: str = None):
    """
    Reads a CSV file from data/raw/csvs/. If filter_column and filter_value are
    given, returns only rows where that column matches (case-insensitive,
    partial match). Otherwise returns all rows.
    """
    rows = _load_csv(filename)
    if isinstance(rows, dict) and "error" in rows:
        return rows

    if filter_column and filter_value:
        filter_value_lower = filter_value.lower()
        matches = [
            r for r in rows
            if filter_column in r and filter_value_lower in str(r[filter_column]).lower()
        ]
        return matches if matches else {"error": f"No rows found where {filter_column} matches '{filter_value}'"}

    return rows


# ---------- Tool schemas (Ollama / OpenAI-style function calling format) ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_application_steps",
            "description": "Get the admission application procedure. Pass a category like 'Punjab' or 'Other States' or 'Lateral Entry' for a specific management-quota procedure, or omit for the general step-by-step process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Optional category filter"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_counselling_schedule",
            "description": "Get IKGPTU centralized counselling schedule. Pass course_type like 'undergraduate', 'pg', or 'lateral' to filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_type": {"type": "string", "description": "Optional filter for programme type"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_hostel_info",
            "description": "Get hostel accommodation details and campus facilities.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_placement_stats",
            "description": "Get placement statistics for a course. Prefer passing course_name (e.g. 'B.Tech CSE') since that's what students say - the tool resolves it to the internal course_id automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_name": {"type": "string", "description": "Course name as the student would say it"},
                    "course_id": {"type": "string", "description": "Internal course_id if already known"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search frequently asked questions by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search for in FAQs"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scholarship_info",
            "description": "Get details about the AGC NEST scholarship scheme.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_anti_ragging_info",
            "description": "Get anti-ragging policy and helpline information. Always use this for any ragging-related question rather than answering from memory.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_grievance_info",
            "description": "Get student grievance redressal committee information. Pass topic='women' for the Women/Anti-Sexual-Harassment committee, otherwise general student grievance committee.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional: 'women' for women's grievance committee"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_admission_office",
            "description": "Get contact info to escalate to a human admission officer. Pass region (e.g. 'Bihar', 'Jammu') if the student mentioned one, otherwise returns the main contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Optional region name"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_courses_by_stream",
            "description": "Get courses suited for a student's 12th-grade stream. Use this whenever a student asks which courses they can take based on their stream (e.g. 'medical stream', 'non-medical', 'PCB', 'commerce', 'arts') rather than trying to figure it out yourself. Returns a ready-formatted 'display_text' - output it directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stream": {"type": "string", "description": "'medical', 'non-medical', 'commerce', or 'arts'"}
                },
                "required": ["stream"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_courses_list",
            "description": "Get ALL courses (name + duration). Returns a 'display_text' field that is ALREADY formatted as a bullet list - when answering, output that display_text directly to the student rather than reformatting it yourself. Use get_courses_by_level instead if the student specifically asks for UG or PG courses only.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_courses_by_level",
            "description": "Get courses filtered to only 'UG' (undergraduate/Bachelor's), 'PG' (postgraduate/Master's), or 'Doctorate'. Use this whenever the student specifically asks for undergraduate or postgraduate courses - the filtering is already done correctly, do not try to separate UG/PG yourself from get_all_courses_list. Returns a ready-formatted 'display_text' field - output it directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "'UG', 'PG', or 'Doctorate'"}
                },
                "required": ["level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_program_summary",
            "description": "Get pre-calculated aggregate stats: total number of programs, total intake/seats across ALL programs, and intake broken down by discipline. ALWAYS use this for any question asking for a total, sum, or count across multiple courses - never try to add up numbers yourself from get_course_info results.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_eligibility_info",
            "description": "Get general subject-stream eligibility norms (e.g. PCM required for B.Tech) by discipline. Does NOT include exact percentage cutoffs - if the student asks for a specific minimum percentage, tell them this isn't available and escalate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "discipline": {"type": "string", "description": "Discipline name, e.g. 'Engineering', 'Pharmacy', 'Law'"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_course_info",
            "description": "Get real course info (discipline, program name, duration, intake/seats) for a course. This is the authoritative course list. Omit course_name to list all 41 courses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_name": {"type": "string", "description": "Course name or abbreviation as the student said it, e.g. 'CSE', 'MBA', 'B.Tech AI'"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_semester_fee_structure",
            "description": "Get the real semester-wise fee breakdown for a course (e.g. 'B.Tech CSE', 'MBA', 'B.Pharmacy'). This is the authoritative fee source. Returns a ready-formatted 'display_text' listing each actual semester's fee - output that directly, do not group or re-summarize the semesters yourself, the number of semesters varies by course and you will invent wrong groupings if you try.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_name": {"type": "string", "description": "Course name as the student said it"}
                },
                "required": ["course_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_csv_files",
            "description": "List which CSV data files are available to query. Call this first if you're unsure which CSV file has the data you need.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_csv",
            "description": "Read data from a CSV file (e.g. courses.csv, fees.csv, seats.csv). Optionally filter by a column name and value to find specific rows, e.g. filter_column='course_name', filter_value='CSE'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "CSV filename, e.g. 'fees.csv'"},
                    "filter_column": {"type": "string", "description": "Optional column name to filter by"},
                    "filter_value": {"type": "string", "description": "Optional value to match in that column"}
                },
                "required": ["filename"]
            }
        }
    },
]

# Maps tool name -> actual Python function, used by the agent loop to execute calls
TOOL_FUNCTIONS = {
    "get_application_steps": get_application_steps,
    "get_counselling_schedule": get_counselling_schedule,
    "get_hostel_info": get_hostel_info,
    "get_placement_stats": get_placement_stats,
    "search_faq": search_faq,
    "get_scholarship_info": get_scholarship_info,
    "get_anti_ragging_info": get_anti_ragging_info,
    "get_grievance_info": get_grievance_info,
    "escalate_to_admission_office": escalate_to_admission_office,
    "get_courses_by_stream": get_courses_by_stream,
    "get_all_courses_list": get_all_courses_list,
    "get_courses_by_level": get_courses_by_level,
    "get_program_summary": get_program_summary,
    "get_eligibility_info": get_eligibility_info,
    "get_course_info": get_course_info,
    "get_semester_fee_structure": get_semester_fee_structure,
    "list_available_csv_files": list_available_csv_files,
    "query_csv": query_csv,
}