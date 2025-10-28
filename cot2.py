# cot_gemini_cot.py
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re
from typing import Any, List, Dict

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are an expert reasoning assistant. For every user problem you MUST produce a chain-of-thought
as an ARRAY of JSON step objects (no other text). The sequence must be:
- One START step (summary of the user's request)
- Zero or more PLAN steps (detailed reasoning steps)
- One OUTPUT step (final answer)

Return exactly a JSON array like:
[
  {"step":"START", "content":"..."},
  {"step":"PLAN",  "content":"..."},
  {"step":"PLAN",  "content":"..."},
  {"step":"OUTPUT","content":"..."}
]

Do NOT return any additional commentary or metadata outside the array.
"""

# Schema: enforce an array of step objects
json_schema = {
    "name": "cot_array_schema",
    "schema": {
        "type": "array",
        "minItems": 2,
        "items": {
            "type": "object",
            "properties": {
                "step": {"type": "string", "enum": ["START", "PLAN", "OUTPUT"]},
                "content": {"type": "string"}
            },
            "required": ["step", "content"],
            "additionalProperties": False
        }
    }
}

def normalize_resp(raw: Any) -> List[Dict[str, str]]:
    """
    Convert response (dict/list/string) into a list of step objects.
    Uses defensive parsing if needed.
    """
    # If already a list of dicts
    if isinstance(raw, list):
        return raw
    # If it's a dict (single object) — wrap it, but we expect an array normally
    if isinstance(raw, dict):
        return [raw]
    # If string — try to json.loads
    if isinstance(raw, str):
        s = raw.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            # Best-effort: attempt to extract first array [...] block
            m = re.search(r'\[(?:[^\[\]]|(?R))*\]', s, re.DOTALL)
            if not m:
                # simple fallback: find first {...} occurrences
                objs = re.findall(r'\{(?:[^{}]|\n)*\}', s, re.DOTALL)
                results = []
                for o in objs:
                    try:
                        results.append(json.loads(o))
                    except Exception:
                        continue
                return results
            else:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
    return []

def pretty_print_steps(steps: List[Dict[str, str]]):
    for obj in steps:
        st = obj.get("step")
        ct = obj.get("content", "")
        if st == "START":
            print("🔥", ct)
        elif st == "PLAN":
            print("🧠", ct)
        elif st == "OUTPUT":
            print("✨", ct)
        else:
            print("❗", obj)

def main():
    # initial conversation
    history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    user_query = input("👍 Enter your query: ").strip()
    if not user_query:
        print("No query provided — exiting.")
        return

    history.append({"role": "user", "content": user_query})

    try:
        # single request should return an array of steps
        resp = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=history,
            response_format={"type": "json_schema", "json_schema": json_schema}
        )

        raw_content = resp.choices[0].message.content

        # normalize to list of step objects
        steps = normalize_resp(raw_content)
        if not steps:
            print("❌ Failed to parse a valid array of step objects from assistant response.")
            print("Raw assistant output:\n", raw_content)
            return

        # print chain-of-thought
        pretty_print_steps(steps)

        # Append assistant output (stringified JSON array) to history for context if you want
        history.append({"role": "assistant", "content": json.dumps(steps)})

    except Exception as e:
        print("Error calling the API:", str(e))

if __name__ == "__main__":
    main()

    # Quick test (uncomment to run a canned test instead of interactive prompt)
    # test_query = "Solve and show chain-of-thought for (25 + 5*2) / 5."
    # history = [{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":test_query}]
    # resp = client.chat.completions.create(
    #     model="gemini-2.5-flash",
    #     messages=history,
    #     response_format={"type":"json_schema","json_schema":json_schema}
    # )
    # print(resp.choices[0].message.content)
