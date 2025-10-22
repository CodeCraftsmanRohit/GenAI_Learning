from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # match env variable name
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
SYSTEM_PROMPT="YOU SHOULD ANSWER ONLY AND ONLY THE CODING RELATED QUESTIONS . DO NOT ANSWER ANYTHING ELSE. YOUR NAME IS ALEXA.IF ASKED SOMETHING ELSE JUST SAY SORRY"

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    reasoning_effort="low",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey , tell me a joke"}
    ]
)

print(response.choices[0].message.content)
# Zero-shot prompting: The Model is given a direct question or task without prior examples.