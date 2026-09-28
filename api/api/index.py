import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(
    title="Agentic RAG + MCP API",
    description="Intelligent Research Assistant for Document Question Answering",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    documents_path = PROJECT_ROOT / "data" / "documents"

    if not documents_path.exists():
        return {
            "documents": []
        }

    files = [
        file.name
        for file in documents_path.glob("*.pdf")
    ]

    return {
        "documents": files
    }