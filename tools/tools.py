"""
tools.py
Defines the actual Python functions the agent can call, and the JSON schemas
that describe them to the LLM (Ollama tool-calling format).
"""

import json
import os

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

def get_important_dates():
    return _load("dates.json")


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
        if val in ("—", "-", "", "N/A"):
            return None
        try:
            return int(val.replace(",", ""))
        except ValueError:
            return val

    query = course_name.lower().strip()
    for prefix in ["b.tech", "btech", "m.tech", "mtech"]:
        query = query.replace(prefix, "").strip()

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

    if not match:
        return {"error": f"No fee data found for course matching '{course_name}'"}

    semester_fees = {}
    for key, val in match.items():
        if "Sem" in key:
            cleaned = clean_number(val)
            if cleaned is not None:
                semester_fees[key] = cleaned

    return {
        "course": match.get("Course"),
        "semester_wise_fees": semester_fees,
        "total_fee": clean_number(match.get("Total")),
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
            "name": "get_important_dates",
            "description": "Get important admission dates like application deadline, exam dates, counseling dates.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
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
            "description": "Get the real semester-wise fee breakdown for a course (e.g. 'B.Tech CSE', 'MBA', 'B.Pharmacy'). This is the authoritative fee source - prefer this over query_csv for fee questions.",
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
    "get_important_dates": get_important_dates,
    "get_application_steps": get_application_steps,
    "get_counselling_schedule": get_counselling_schedule,
    "get_hostel_info": get_hostel_info,
    "get_placement_stats": get_placement_stats,
    "search_faq": search_faq,
    "get_scholarship_info": get_scholarship_info,
    "get_anti_ragging_info": get_anti_ragging_info,
    "get_grievance_info": get_grievance_info,
    "escalate_to_admission_office": escalate_to_admission_office,
    "get_course_info": get_course_info,
    "get_semester_fee_structure": get_semester_fee_structure,
    "list_available_csv_files": list_available_csv_files,
    "query_csv": query_csv,
}