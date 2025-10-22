from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """YOU SHOULD ANSWER ONLY AND ONLY THE CODING RELATED QUESTIONS . DO NOT ANSWER ANYTHING ELSE. YOUR NAME IS ALEXA. IF ASKED SOMETHING ELSE SAY sorry like this

RULE:
1.> return answer in JSON format only
{{
    "code":"string" or null,
    "isCodingQuestion":boolean
}}

Examples:
Q: Can you explain the a+b whole square?
A:{{
    "code":null,
    "isCodingQuestion":true
}}

Q: write code in python for printing addition of two numbers.
A:{{
    "code":"def add(a,b):\\n    return a+b",
    "isCodingQuestion":false
}}
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    reasoning_effort="low",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Give me code to find factorial"}
    ]
)

# ✅ Clean JSON output (no markdown)
raw_output = response.choices[0].message.content.strip()

# remove ```json or ``` if the model wraps it
if raw_output.startswith("```"):
    raw_output = raw_output.strip("```json").strip("```").strip()

# parse and reprint clean formatted JSON
try:
    parsed = json.loads(raw_output)
    print(json.dumps(parsed, indent=4))
except json.JSONDecodeError:
    print(raw_output)
