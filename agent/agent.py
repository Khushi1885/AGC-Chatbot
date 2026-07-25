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

SYSTEM_PROMPT = """You are the admission cell assistant for Amritsar Group of Colleges (AGC).
Answer student questions about courses, fees, eligibility, admission process, counselling
schedules, hostel, placements, scholarships, and grievance/ragging support.

Rules:
- ALWAYS use the provided tools to look up facts (fees, dates, contacts, procedures).
  Never guess or make up numbers, dates, or contact details.
- For course questions, use get_course_info. For fee questions, use
  get_semester_fee_structure. Both accept course_name directly (e.g. "B.Tech CSE",
  "MBA") - never ask the student for an internal course_id, they won't know it.
- If a tool returns an error or no match, tell the student honestly and offer to
  escalate to the admission office instead of guessing.
- For ragging or grievance related questions, always call the relevant tool and give
  the exact contact info returned.
- Keep answers concise and student-friendly.
"""


def run_agent(user_message: str, conversation_history: list = None):
    """
    Runs one turn of the agent loop for a given user message.
    conversation_history: list of prior {"role": ..., "content": ...} messages (optional)
    Returns the final text response from the agent.
    """
    messages = conversation_history[:] if conversation_history else []
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})

    max_iterations = 6  # safety limit to avoid infinite tool-call loops

    for _ in range(max_iterations):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        message = result.get("message", {})

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            # Final answer - no more tools needed
            messages.append(message)
            return message.get("content", ""), messages

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

            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    return "Sorry, I couldn't process that after several attempts. Please contact the admission office directly.", messages


if __name__ == "__main__":
    # Quick manual test from the command line
    print("AGC Admission Assistant (type 'quit' to exit)\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        answer, history = run_agent(user_input, history)
        print(f"\nAgent: {answer}\n")