from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINIAI_API_KEY = os.getenv("GEMINIAI_API_KEY")

client = genai.Client(
    api_key=GEMINIAI_API_KEY
)

app = FastAPI(
    title="Agentic RAG + MCP API",
    description="Intelligent Research Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    document_name: str = ""


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Agentic RAG + MCP API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/api/documents")
def documents():
    documents_path = Path("data/documents")

    if not documents_path.exists():
        return {"documents": []}

    files = [
        file.name
        for file in documents_path.glob("*.pdf")
    ]

    return {"documents": files}


@app.post("/api/ask")
def ask(request: AskRequest):

    prompt = f"""
You are an intelligent research assistant.

User Question:
{request.question}

Selected Document:
{request.document_name}

Answer clearly and concisely.

Important:
If you do not have enough document context to answer,
say that the information is not available.
Do not invent information.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {
        "question": request.question,
        "document": request.document_name,
        "answer": response.text,
        "sources": []
    }