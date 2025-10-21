from google import genai

client=genai.Client(
    api_key="AIzaSyCAAXF3aJD8ywcrMSnkDXLmsX185axXqlc"
)

response=client.models.generate_content(
     model="gemini-2.5-flash",contents="Explain how AI works in a few words"
)
print(response.text)