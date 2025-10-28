from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)



SYSTEM_PROMPT = """
You're an expert AI Assistant that answers using explicit steps in JSON.
Follow the sequence: START, multiple PLAN entries (optional), then OUTPUT.
Return only JSON in this exact shape for each message:
{"step":"START" | "PLAN" | "OUTPUT", "content":"string"}

Example (valid JSON messages):
{"step":"START","content":"Hey, Can you solve 2+3*5/10"}
{"step":"PLAN","content":"User is asking for arithmetic; apply BODMAS"}
{"step":"PLAN","content":"Compute 3*5 = 15"}
{"step":"PLAN","content":"Compute 15/10 = 1.5"}
{"step":"PLAN","content":"Compute 2 + 1.5 = 3.5"}
{"step":"OUTPUT","content":"3.5"}
"""


message_history=[
    {"role":"system"."content":SYSTEM_PROMPT},

]

user_query=input("👍")
message_history.append({"role":"user","content":user_query})


while True:
    response=client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type":"json_object"},
        messages=message_history
    )
    raw_result=response.choices[0].message.content
    message_history.append({"role":"assistant","content":raw_result})
    parsed_result=json.loads(raw_result)
    if parsed_result.get("step") == "START":
    print("🔥", parsed_result.get("content"))
    continue

    if parsed_result.get("step") == "PLAN":
    print("🧠", parsed_result.get("content"))
    continue

    if parsed_result.get("step") == "OUTPUT":
    print("✨", parsed_result.get("content"))
    continue


response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type":"json_object"},
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey, write a code to add n numbers in js"}
    ],

)
