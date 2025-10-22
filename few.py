from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # match env variable name
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
SYSTEM_PROMPT="""YOU SHOULD ANSWER ONLY AND ONLY THE CODING RELATED QUESTIONS . DO NOT ANSWER ANYTHING ELSE. YOUR NAME IS ALEXA.IF ASKED SOMETHING ELSE SAY soory like this

Examples:
Q: Can you explain the a+b whole square?
A:sorry , i can answer only coding related questions


"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    reasoning_effort="low",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey , tell me a joke"}
    ]
)

print(response.choices[0].message.content)
# few shot prompting -> directly giving the inst. to the model and few examples to the model