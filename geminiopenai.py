from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # match env variable name
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    reasoning_effort="high",
    messages=[
        {"role": "system", "content": "You are an expert in Maths and only answer Maths related questions.that if query is not related to math just say sorry"},
        {"role": "user", "content": "Hey , solve a+b square"}
    ]
)

print(response.choices[0].message.content)
