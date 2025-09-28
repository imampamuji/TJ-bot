from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import os
import getpass

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    
# Language model   
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")