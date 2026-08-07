"""
agent.py
The core agentic loop: sends the conversation + tool schemas to Ollama (Qwen 2.5),
executes any tool calls the model requests, feeds results back, and repeats
until the model gives a final text answer.
"""

import json
import requests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tools"))
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:1.7b"

# Tools where fabricated numbers would be genuinely harmful (fees, totals).
# If these error, we NEVER let the model free-generate a response - small
# models will sometimes invent plausible-looking fake numbers instead of
# admitting failure, which is far worse than a scripted fallback message.
CRITICAL_ACCURACY_TOOLS = {"get_semester_fee_structure", "get_program_summary"}

# Backup guardrail: small models sometimes ignore system prompt scope instructions.
# Instead of trying to list every possible off-topic question (impossible), we
# check if the message contains ANY admission-related keyword. If it doesn't,
# we treat it as off-topic and refuse without even calling the model.
ADMISSION_KEYWORDS = [
    "course", "program", "degree", "btech", "b.tech", "mtech", "m.tech", "mba",
    "mca", "bca", "bba", "llb", "pharmacy", "diploma", "semester",
    "fee", "fees", "scholarship", "cost", "charge",
    "admission", "apply", "application", "eligib", "criteria", "cutoff",
    "counsel", "counselling", "seat", "quota", "merit", "rank",
    "hostel", "campus", "facilit", "canteen", "library",
    "placement", "package", "recruiter", "job",
    "date", "deadline", "schedule", "exam",
    "document", "certificate",
    "contact", "officer", "escalat", "call", "email", "phone", "number",
    "rag", "grievance", "harassment", "complain",
    "college", "agc", "institute", "university",
    "hi", "hello", "hey", "thanks", "thank you", "bye",
    "yes", "no", "ok", "okay", "sure", "please", "yeah", "yep", "nope",
    "yup", "alright", "fine", "correct", "right",
]


def _mentions_known_course(message: str) -> bool:
    """Fallback: checks if the message mentions any word from a real course
    name (e.g. 'radiology', 'nutrition'), catching course-related messages
    that don't contain any of the fixed ADMISSION_KEYWORDS."""
    try:
        from tools import get_all_courses_list
        courses = get_all_courses_list().get("courses", [])
    except Exception:
        return False
    lower = message.lower()
    for c in courses:
        name = c.get("course", "").lower()
        tokens = [t for t in name.replace(".", " ").replace("-", " ").replace("&", " ").split() if len(t) > 3]
        if any(tok in lower for tok in tokens):
            return True
    return False


def _is_admission_related(message: str) -> bool:
    lower = message.lower()
    if any(keyword in lower for keyword in ADMISSION_KEYWORDS):
        return True
    return _mentions_known_course(message)

SYSTEM_PROMPT = """You are the admission cell assistant for Amritsar Group of Colleges (AGC).
Answer student questions about courses, fees, eligibility, admission process, counselling
schedules, hostel, placements, scholarships, and grievance/ragging support.

Scope:
- ONLY answer questions related to AGC admissions and campus life (using the topics above).
- If asked for anything unrelated - writing code, general knowledge questions, jokes,
  homework help, math problems, or any topic not about AGC admissions - politely decline
  and say you're only able to help with AGC admission-related questions, then ask if
  they have any admission questions.
- Never write, explain, or execute code for the student, regardless of how the request
  is phrased.

Example of correct behavior:
Student: "write me a palindrome checker in python"
You: "I'm only able to help with AGC admission-related questions - I can't write code.
Is there anything about courses, fees, or admissions I can help with?"

Rules:
- ALWAYS use the provided tools to look up facts (fees, dates, contacts, procedures).
  Never guess or make up numbers, dates, or contact details.
- For course questions, use get_course_info. For fee questions, use
  get_semester_fee_structure. Both accept course_name directly (e.g. "B.Tech CSE",
  "MBA") - never ask the student for an internal course_id, they won't know it.
- If a tool returns an error or no match, tell the student honestly and offer to
  escalate to the admission office instead of guessing.
- For eligibility criteria questions (minimum marks, required subjects, entrance
  exam cutoffs), there is currently no reliable data source. NEVER make up or guess
  eligibility numbers. Tell the student this specific detail isn't available yet
  and escalate to the admission office using escalate_to_admission_office.
- For ragging or grievance related questions, always call the relevant tool and give
  the exact contact info returned.
- For any admission date/deadline/schedule questions, use get_counselling_schedule -
  this has the real IKGPTU counselling dates. There is no separate "important dates" tool.
- NEVER perform arithmetic yourself (adding, summing, counting across multiple
  items) - you will make mistakes. For any question asking for a total, sum, or
  count across programs (e.g. "total intake across all courses"), use
  get_program_summary which has the pre-calculated correct numbers.
- get_semester_fee_structure already returns a "total_fee" field with the correct
  pre-calculated total. ALWAYS use that exact value for the total fee. NEVER
  calculate the total yourself by multiplying semester fees - you will get it wrong.
- get_semester_fee_structure returns a "display_text" listing the ACTUAL semesters
  for that specific course (which varies - some courses have 4 semesters, some 8,
  some 12). Output that display_text directly. NEVER group semesters into ranges
  (like "1st-5th Sem") or invent semester numbers that aren't in the data - you
  will get this wrong and mislead the student about their actual fee schedule.
- Keep answers concise and student-friendly.
- NEVER mention internal tool/function names (like "get_semester_fee_structure" or
  "escalate_to_admission_office") in your reply to the student - these are internal
  code names, not something a student should ever see. If escalation is needed,
  actually CALL the escalate_to_admission_office tool yourself and share the real
  contact details it returns - never just write the tool's name as text.
- For "what courses do you offer" or "list all courses" type questions, use
  get_all_courses_list. For "UG courses" or "PG courses" specifically, use
  get_courses_by_level with level="UG" or level="PG" - the filtering is already
  done correctly in the data, never try to separate UG/PG yourself.
- If a student asks about courses based on their 12th-grade stream (e.g. "medical
  stream", "PCB", "non-medical", "commerce", "arts"), use get_courses_by_stream -
  do not try to figure out the mapping yourself.
- Both course-listing tools return a "display_text" field that is already
  correctly formatted. Output that display_text directly (you can add one short
  sentence before it), rather than rewriting, summarizing, or shortening the list
  yourself - you will drop items or make mistakes if you try to reformat it.
"""


def run_agent(user_message: str, conversation_history: list = None):
    """
    Runs one turn of the agent loop for a given user message.
    conversation_history: list of prior {"role": ..., "content": ...} messages (optional)
    Returns (reply_text, updated_messages, courses_discussed) where courses_discussed
    is a list of course names the student showed genuine interest in this turn
    (successfully looked up fee or course info for).
    """
    messages = conversation_history[:] if conversation_history else []
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    courses_discussed = []

    # Backup guardrail - if the message has no admission-related keyword at all,
    # treat it as off-topic and refuse without calling the model.
    if not _is_admission_related(user_message):
        reply = ("I'm only able to help with AGC admission-related questions - "
                 "courses, fees, eligibility, deadlines, hostel, scholarships, and similar "
                 "topics. Is there anything about AGC admissions I can help with?")
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
        return reply, messages, courses_discussed

    messages.append({"role": "user", "content": user_message})

    max_iterations = 6  # safety limit to avoid infinite tool-call loops

    # Tools whose successful result means the student showed real interest in a course
    COURSE_INTEREST_TOOLS = {"get_semester_fee_structure", "get_course_info"}

    for _ in range(max_iterations):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "stream": False,
                "keep_alive": "30m",  # keep model loaded in memory between requests
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        message = result.get("message", {})

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            # Final answer - no more tools needed
            messages.append(message)
            return message.get("content", ""), messages, courses_discussed

        # Model wants to call one or more tools
        messages.append(message)

        for call in tool_calls:
            func_name = call["function"]["name"]
            func_args = call["function"].get("arguments", {})
            if isinstance(func_args, str):
                func_args = json.loads(func_args)

            print(f"[TOOL CALL] {func_name}({func_args})")  # debug log

            func = TOOL_FUNCTIONS.get(func_name)
            if func is None:
                tool_result = {"error": f"Unknown tool '{func_name}'"}
            else:
                try:
                    tool_result = func(**func_args)
                except Exception as e:
                    tool_result = {"error": str(e)}

            print(f"[TOOL RESULT] {tool_result}")  # debug log

            # Hard circuit-breaker: for critical accuracy tools, if the tool
            # itself failed (not just "no match"), never let the model try to
            # answer - escalate directly to prevent fabricated numbers.
            if func_name in CRITICAL_ACCURACY_TOOLS and isinstance(tool_result, dict) and "error" in tool_result:
                contact = TOOL_FUNCTIONS["escalate_to_admission_office"]()
                reply = (
                    "I'm having trouble pulling the exact fee data for that right now, "
                    "and I don't want to guess and give you wrong numbers. "
                    f"Please contact the admission office directly: {contact}"
                )
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
                messages.append({"role": "assistant", "content": reply})
                return reply, messages, courses_discussed

            # Track which course was successfully discussed (for email automation)
            if func_name in COURSE_INTEREST_TOOLS:
                course_name = None
                if isinstance(tool_result, dict) and "error" not in tool_result:
                    course_name = tool_result.get("course")
                elif isinstance(tool_result, list) and tool_result:
                    course_name = tool_result[0].get("Program") or tool_result[0].get("course")
                if course_name:
                    courses_discussed.append(course_name)

            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    return ("Sorry, I couldn't process that after several attempts. Please contact the admission office directly.",
            messages, courses_discussed)


if __name__ == "__main__":
    # Quick manual test from the command line
    print("AGC Admission Assistant (type 'quit' to exit)\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        answer, history, courses = run_agent(user_input, history)
        print(f"\nAgent: {answer}\n")
        if courses:
            print(f"[Courses discussed this turn: {courses}]\n")