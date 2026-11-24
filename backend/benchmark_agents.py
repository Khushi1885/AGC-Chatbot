"""
benchmark_agents.py - measures how well YOUR local model picks tools and fills arguments.

Run it from the backend folder (Ollama must be running):

    python benchmark_agents.py                       # uses the model from agent.py
    AGENT_MODEL=qwen2.5:3b python benchmark_agents.py
    AGENT_MODEL=qwen3:4b   python benchmark_agents.py

(Windows PowerShell:  $env:AGENT_MODEL="qwen2.5:3b"; python benchmark_agents.py)

It only tests the FIRST tool call of each agent - the step small models get wrong most
often - so it needs no database. Pick the smallest model that scores >= 80%, since the
real system also has Python guardrails and fallbacks on top of this.
"""
import sys, time, types
from datetime import date

try:                                    # the benchmark never touches the database
    import db  # noqa: F401
except Exception:
    stub = types.ModuleType("db"); stub.supabase = None; sys.modules["db"] = stub

import agent_runtime as rt
import agents
import admin_agent

agent_runtime_persist = False
rt.PERSIST_TRACES = False
TODAY = date.today()


def first_call(system, user, schemas):
    t0 = time.time()
    msg = rt.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                  schemas, timeout=90, lock_wait=5, max_tokens=140)
    calls = msg.get("tool_calls") or []
    took = time.time() - t0
    if not calls:
        return None, {}, took
    fn = calls[0]["function"]
    return fn.get("name"), rt._as_dict(fn.get("arguments")), took


def schema(name, desc, params):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


SLOT_SCHEMAS = [
    schema("search_slots", "Search the counsellor's free slots.", agents.SLOT_SEARCH_PARAMS),
    schema("offer_slots", "Offer up to 4 of the found slots.", agents.SLOT_OFFER_PARAMS),
    schema("no_match", "Use when no slot can satisfy the student.", agents.SLOT_NONE_PARAMS),
]
SLOT_CASES = [  # message, expected exclude, expected prefer, expected part, expected next_week
    ("Friday ko possible nahi", [4], [], None, False),
    ("Tuesday afternoon", [], [1], "afternoon", False),
    ("Monday nahi, Tuesday theek hai", [0], [1], None, False),
    ("next week morning", [], [], "morning", True),
    ("I can't make Wednesday but Thursday morning works", [2], [3], "morning", False),
    ("shukravar ko nahi, subah chahiye", [4], [], "morning", False),
    ("any afternoon works for me", [], [], "afternoon", False),
    ("Monday or Tuesday please", [], [0, 1], None, False),
]

ADMIN_SCHEMAS = [
    schema("find_students", "List students, optionally filtered.", admin_agent.FIND_PARAMS),
    schema("counsellor_workload", "Each counsellor's student counts.", admin_agent.EMPTY_PARAMS),
    schema("lead_summary", "Overall counts by lead label, status and source.", admin_agent.SUMMARY_PARAMS),
]
ADMIN_CASES = [  # question, expected tool, expected args (subset)
    ("Hot leads from this week that haven't been contacted yet", "find_students", {"label": "Hot", "status": "New"}),
    ("workload per counsellor", "counsellor_workload", {}),
    ("summary of last 30 days", "lead_summary", {"days": 30}),
    ("BCA students not contacted", "find_students", {"course_contains": "BCA", "status": "New"}),
    ("Warm leads from AGC NEST", "find_students", {"label": "Warm", "source": "nest"}),
    ("Hot leads ko follow up bhejo", "find_students", {"label": "Hot"}),
]

ALLOC_SCHEMA = [schema("assign", "Assign the student to one counsellor.", agents.ALLOC_PARAMS)]
ALLOC_CASES = [  # student line, candidates, expected id
    ("course=B.Tech CSE (discipline: Engineering), lead=Hot",
     ["id=1 name=Rahul specializations=Engineering open_students=2 hot_leads=1",
      "id=3 name=Aman specializations=Engineering open_students=2 hot_leads=0"], 3),
    ("course=B.Pharmacy (discipline: Pharmacy), lead=Warm",
     ["id=4 name=Priya specializations=Pharmacy open_students=3 hot_leads=0",
      "id=5 name=Gurpreet specializations=Computer Applications,Management open_students=2 hot_leads=0"], 4),
    ("course=MBA (discipline: Management), lead=Warm",
     ["id=2 name=Simran specializations=Management open_students=1 hot_leads=0",
      "id=5 name=Gurpreet specializations=Computer Applications,Management open_students=2 hot_leads=0"], 2),
]


def norm_part(v):
    return v if v in ("morning", "afternoon") else None


def main():
    _, model = rt.llm_config()
    print(f"\nModel: {model}\n")
    if not rt.llm_available():
        print("Agents are disabled (AGENTS_ENABLED=0)."); return
    results = {"scheduling": [], "admin": [], "allocation": []}
    times = []

    print("SCHEDULING agent - first tool call")
    for text, ex, pr, part, nxt in SLOT_CASES:
        try:
            name, a, took = first_call(agents.slot_system(TODAY), text, SLOT_SCHEMAS)
            got_ex = sorted(agents._weekday_list(a.get("exclude_weekdays"))) if name else None
            got_pr = sorted(agents._weekday_list(a.get("prefer_weekdays"))) if name else None
            ok = (name == "search_slots" and got_ex == ex and got_pr == pr
                  and norm_part(a.get("part_of_day")) == part
                  and bool(a.get("next_week") in (True, "true", "True", 1)) == nxt)
        except Exception as e:
            ok, took, name = False, 0, f"error: {e}"
        results["scheduling"].append(ok); times.append(took)
        print(f"  {'PASS' if ok else 'FAIL'}  {took:5.1f}s  {text!r}")

    print("\nADMIN agent - first tool call")
    for q, tool, exp in ADMIN_CASES:
        try:
            name, a, took = first_call(admin_agent.SYSTEM, q, ADMIN_SCHEMAS)
            clean = admin_agent.sanitize_args(a)
            ok = name == tool and all(str(clean.get(k, "")).lower() == str(v).lower() for k, v in exp.items())
        except Exception as e:
            ok, took = False, 0
        results["admin"].append(ok); times.append(took)
        print(f"  {'PASS' if ok else 'FAIL'}  {took:5.1f}s  {q!r}")

    print("\nALLOCATION agent - first tool call")
    for student, cands, expected in ALLOC_CASES:
        try:
            name, a, took = first_call(agents.ALLOC_SYSTEM, f"Student: {student}.\nCandidates:\n" + "\n".join(cands), ALLOC_SCHEMA)
            ok = name == "assign" and int(a.get("counsellor_id", -1)) == expected
        except Exception as e:
            ok, took = False, 0
        results["allocation"].append(ok); times.append(took)
        print(f"  {'PASS' if ok else 'FAIL'}  {took:5.1f}s  {student}")

    print("\n" + "=" * 52)
    total = ok_total = 0
    for k, v in results.items():
        total += len(v); ok_total += sum(v)
        print(f"  {k:<11} {sum(v)}/{len(v)}  ({100 * sum(v) // max(len(v), 1)}%)")
    avg = sum(times) / max(len(times), 1)
    print(f"  OVERALL     {ok_total}/{total}  ({100 * ok_total // max(total, 1)}%)   average {avg:.1f}s per call")
    print("=" * 52)
    print("Guide: >= 80% -> use this model.  < 80% -> try a bigger one (the guardrails still protect you,")
    print("but fallbacks will fire more often).  Average > 20s -> too slow on this laptop; try a smaller one.")


if __name__ == "__main__":
    main()