from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

# Load environment variables (make sure OPENAI_API_KEY is set in .env)
load_dotenv()

# Path to your PDF file (same directory as script)
pdf_path = Path(__file__).parent / "test.pdf"

# 1️⃣ Load the PDF
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# 2️⃣ Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)

# 3️⃣ Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("OPENAI_API_KEY")
)
# 4️⃣ Create Qdrant Vector Store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("✅ Indexing of documents done successfully!")
