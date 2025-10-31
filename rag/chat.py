from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
load_dotenv()
import os

# Vector Embeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("OPENAI_API_KEY")
)

vector_db=QdrantVectorStore.from_existing_collection(

    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

user_query=input("Ask something: ")

search_results=vector_db.similarity_search(query=user_query)

context = "\n\n\n".join(f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
for result in search_results)


SYSTEM_PROMPT = f"""
You are a helpfull AI Assistant who answeres user query based on the available
context retrieved from a PDF file along with page_contents and page number.

You should only ans the user based on the following context and navigate the
user to open the right page number to know more.

Context:
{context}
"""


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # match env variable name
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    reasoning_effort="high",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

print(f"🔥:{response.choices[0].message.content}")